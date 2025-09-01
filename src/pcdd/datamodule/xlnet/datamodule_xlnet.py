from typing import List, Literal, Optional, TypedDict, Union, cast
import torch
from torch import Tensor as TT
from pcdd.datamodule.datamodule import (
    BaseBatch,
    BaseCollatorInput,
    Integer,
    pad_left_truncate,
    pad_prefix_suffix,
    pad_prefix_suffix2,
)
from pcdd.utils.rank_zero import RankedLogger
from pcdd import flags

logger = RankedLogger(__name__, rank_zero_only=True)


class XLNetBaseBatch(TypedDict):
    """
    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]


class XLNetTrainingBatch(XLNetBaseBatch):
    """
    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        perm_mask (Integer[TT, " batch seq_len seq_len"]): 1 for tokens that are allowed to attend to each other.
            - perm_mask[k, i, j] = 0, i attend to j in batch k; This is the opposite of attention_mask.
        target_mapping (Integer[TT, " batch num_predict seq_len"]): Mask to indicate the output tokens to use.
            - If target_mapping[k, i, j] = 1, the i-th predict in batch k is on the j-th token. Only used during pretraining for partial prediction or for sequential decoding (generation).
        token_type_ids (Optional[Integer[TT, " batch seq_len"]]): # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
    """

    perm_mask: Integer[TT, " batch seq_len seq_len"]
    token_type_ids: Optional[Integer[TT, " batch seq_len"]]
    target_mapping: Optional[Integer[TT, " batch num_predict seq_len"]]
    labels: Optional[Integer[TT, " batch num_predict"]]


class XLNetPredictionBatch(XLNetBaseBatch):
    """
    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    drop: Integer[TT, " batch seq_len"]


def xlnet_training_collator(
    examples: List[BaseCollatorInput],
    max_len: int,
    pad_token_id: int,
    prefix_attention_type: Literal["full", "causal"],
) -> XLNetTrainingBatch:
    batch = cast(
        BaseBatch,
        pad_left_truncate(
            examples,
            max_len=max_len,
            type_extension=0,
            pad_token_id=pad_token_id,
            return_tensors=True,
        ),
    )
    input_ids: Integer[TT, " batch seq_len"] = batch["input_ids"]
    attention_mask: Integer[TT, " batch seq_len"] = batch["attention_mask"]
    token_type_ids: Integer[TT, " batch seq_len"] = batch["token_type_ids"]
    max_seq_len = input_ids.shape[-1]
    # random permutation except the prefix
    prefix_mask = token_type_ids <= 1
    num_predict = (~prefix_mask).sum(dim=-1)  # (batch,)
    perm_masks = []
    target_mappings = []
    labels = []
    max_num_predict = 0
    for i, _num_predict in enumerate(num_predict):
        _num_predict = int(_num_predict)
        max_num_predict = max(max_num_predict, _num_predict)
        prefix_length = max_seq_len - _num_predict  # includes left pads

        if prefix_attention_type == "full":
            if not flags.DEBUG_XLNET_AR:
                output_perm = torch.randperm(_num_predict) + 1
            else:
                output_perm = torch.arange(_num_predict) + 1
            position_to_order = torch.zeros(max_seq_len, dtype=torch.long)
        elif prefix_attention_type == "causal":
            output_perm = torch.randperm(_num_predict) + prefix_length
            position_to_order = torch.arange(max_seq_len, dtype=torch.long)
        else:
            raise ValueError(
                f"Invalid prefix attention type: {prefix_attention_type}"
            )

        position_to_order[prefix_length:] = output_perm
        # perm_mask[q,k] = 0 => q attends to k
        # so perm_mask[q,k] should be 0 if position_to_order[k] < position_to_order[q]
        # Therefore, perm_mask[q,k] = position_to_order[k] >= position_to_order[q]
        # Note: position_to_order.unsqueeze(0)[q,k] = position_to_order[k]
        _perm_mask = position_to_order.unsqueeze(
            0
        ) >= position_to_order.unsqueeze(1)
        perm_masks.append(_perm_mask)
        order_to_position = torch.argsort(
            position_to_order, dim=-1
        )  # (seq_len,)
        target_mapping = torch.zeros(
            _num_predict, max_seq_len, dtype=torch.long
        )  # (num_predict, seq_len)
        target_mapping[
            torch.arange(_num_predict), order_to_position[-_num_predict:]
        ] = 1
        target_mappings.append(target_mapping)
        _labels = input_ids[
            i, order_to_position[-_num_predict:]
        ]  # (num_predict,)
        labels.append(_labels)
    # pad labels and target_mappings
    padded_labels = []
    padded_target_mappings = []
    for _labels, _target_mapping in zip(labels, target_mappings):
        num_padding = max_num_predict - len(_labels)
        # padding starts from the last dimension
        _labels = torch.nn.functional.pad(
            _labels,
            (0, num_padding),  # zero left pad
            value=-100,
        )
        # target_mapping will have shape (batch, max_num_predict, seq_len)
        _target_mapping = torch.nn.functional.pad(
            _target_mapping,
            (
                0,
                0,
                0,
                num_padding,
            ),  # (dim1_left, dim1_right, dim0_left, dim0_right)
        )
        padded_labels.append(_labels)
        padded_target_mappings.append(_target_mapping)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "perm_mask": torch.stack(perm_masks, dim=0),
        "target_mapping": torch.stack(padded_target_mappings, dim=0),
        "token_type_ids": token_type_ids,
        "labels": torch.stack(padded_labels, dim=0),
    }


def xlnet_prediction_collator(
    examples: List[BaseCollatorInput],
    max_len: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> XLNetPredictionBatch:
    batch = cast(
        BaseBatch,
        pad_prefix_suffix2(
            examples,
            max_seq_len=max_len,  # not used
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            prefix_type_extension=0,
            suffix_type_extension=2,
            return_tensors=True,
        ),
    )
    drop = batch["token_type_ids"] > 1
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "drop": drop,
    }


def print_batch_xlnet(
    datamodule,
    batch: Union[XLNetTrainingBatch, XLNetPredictionBatch],
    split: Literal["train", "val", "test"],
    dataloader_idx: Optional[int] = None,
):
    self = datamodule
    dataloader_name = ""
    if dataloader_idx is not None:
        dataloader_name = (
            self.train_dataloader_names.get(dataloader_idx, "")
            if split == "train"
            else (
                self.val_dataloader_names.get(dataloader_idx, "")
                if split == "val"
                else self.test_dataloader_names.get(dataloader_idx, "")
            )
        )
    if split in ["train", "val", "test"]:
        logger.info(
            f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
        )
        print("input tokens:")
        print(self.tokenizer.decode(batch["input_ids"][0]))
        print("input_ids:")
        print(batch["input_ids"][0])
        print("attention_mask:")
        print(batch["attention_mask"][0])
        print("drop:")
        print(batch["drop"][0] if "drop" in batch else None)
        print("perm_mask:")
        print(batch["perm_mask"][0] if "perm_mask" in batch else None)
        print("target_mapping:")
        print(
            batch["target_mapping"][0] if "target_mapping" in batch else None
        )
        print("labels:")
        print(batch["labels"][0] if "labels" in batch else None)
