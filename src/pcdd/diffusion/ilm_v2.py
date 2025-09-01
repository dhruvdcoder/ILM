from functools import partial
from itertools import cycle
import json
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import hydra
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from torchmetrics import (
    MeanMetric,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from torchmetrics.classification import BinaryAccuracy

from pcdd import flags
from .nn import (
    general_sample_over_last_two_dims,
    masked_ce_last_two_dims,
    remove_tokens,
)

from pcdd.datamodule.datamodule import (
    ILMBatch,
    Tokenizer,
)
from .lightning_module_v2 import (
    BaseLightningModule,
    LossFunction,
    Predictor,
)
from pcdd.utils.rank_zero import rank_zero_only, RankedLogger
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
)

logger = RankedLogger(__name__, rank_zero_only=True)

################################################################################
# region: Types


class ILMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        batch_loss (Float[TT, " batch"]): Loss value for each example in the batch.
    """

    loss: Float[TT, ""]
    batch_loss: Float[TT, " batch"]


class ILMWithLengthClassificationLossDict(ILMLossDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        per_example_length_loss (Float[TT, " batch"]): Length loss value for each example in the batch.
        per_example_ce (Float[TT, " batch"]): Token prediction loss value for each example in the batch.
        batch_loss (Float[TT, ""]): Combined loss value (length and token prediction) for each example in the batch.
        length_logits (Float[TT, " batch max_length"]): The length logits for each example in the batch.
        n_drops (Integer[TT, " batch"]): The number of tokens dropped for each example in the batch.
    """

    loss: Float[TT, ""]
    per_example_length_loss: Float[TT, " batch"]
    per_example_ce: Float[TT, " batch"]
    batch_loss: Float[TT, ""]
    length_logits: Float[TT, " batch max_length"]
    n_drops: Integer[TT, " batch"]


class ILMWithStoppingClassificationLossDict(ILMLossDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
        per_example_stopping_loss (Float[TT, " batch"]): Stopping loss value for each example in the batch.
        per_example_ce (Float[TT, " batch"]): Token prediction loss value for each example in the batch.
        batch_loss (Float[TT, ""]): Combined loss value (stopping and token prediction) for each example in the batch.
        stopping_logits (Float[TT, " batch 2"]): The stopping logits for each example in the batch.
        n_drops (Integer[TT, " batch"]): The number of tokens dropped for each example in the batch.
    """

    loss: Float[TT, ""]
    per_example_stopping_loss: Float[TT, " batch"]
    per_example_ce: Float[TT, " batch"]
    batch_loss: Float[TT, ""]
    stopping_logits: Float[TT, " batch 2"]
    n_drops: Integer[TT, " batch"]


class ILMPredictionDict(TypedDict):
    """Output of the Predictor for ILM.

    Attributes:
        loss (Optional[Float[TT, "batch"]]): The loss value. Typically None.
        text (List[str]): The batch of generated text without special tokens.
        text_with_spl_tokens (List[str]): The batch of generated text with special tokens.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
        attention_mask (Bool[TT, " batch seq_len"]): Attention mask accompanying the generated ids.
        positions (Integer[TT, " batch seq_len"]): The batch of positions of the generated tokens accompanying the ids.
        history (List[List[Tuple[str, float, int]]]): The batch of history.
            Each entry is a list of tuples, where each tuple contains
            (current_string, time, step_number) of when some change is made to the generated string.
    """

    loss: Optional[Float[TT, ""]]
    text: List[str]
    text_with_spl_tokens: List[str]
    ids: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    positions: Integer[TT, " batch seq_len"]
    history: List[List[Tuple[str, float, int]]]
    time_taken: List[float]


TokenLogitsType = Float[TT, " batch seq_len vocab_size"]
LengthLogitsType = Float[TT, " batch max_length"]
ClassificationLogitsType = Float[TT, " batch num_classes"]


class ILMModel(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        non_drop_non_pad: Integer[TT, " batch seq_len"],
        positions: Integer[TT, " batch seq_len"],
        token_type_ids: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> Tuple[TokenLogitsType, None]: ...


class ILMModelWithStoppingClassification(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        non_drop_non_pad: Integer[TT, " batch seq_len"],
        positions: Integer[TT, " batch seq_len"],
        token_type_ids: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> Tuple[TokenLogitsType, ClassificationLogitsType]: ...


class ILMModelWithLengthClassification(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        non_drop_non_pad: Integer[TT, " batch seq_len"],
        positions: Integer[TT, " batch seq_len"],
        token_type_ids: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> Tuple[TokenLogitsType, LengthLogitsType]: ...

    def get_mean_delta_l(
        self, logits: ClassificationLogitsType
    ) -> Float[TT, " batch"]: ...


# endregion
################################################################################


################################################################################
# region: Base ILM

###############################################################
# region: Loss functions


class ILMLossWithMaskedCE(LossFunction[ILMBatch, ILMLossDict]):
    def __init__(
        self,
        loss_on_padding: bool = False,
        model: Optional[ILMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_constraint: bool = False,
        input_constraint: bool = False,
    ):
        self.loss_on_padding = loss_on_padding
        self.model = model
        self.tokenizer = tokenizer  # type: ignore
        self.use_constraint = use_constraint
        self._min_value: Optional[float] = None
        self.mask_token_id_tensor = None
        self.input_constraint = input_constraint
        if not self.loss_on_padding:
            logger.warning(
                "loss_on_padding is False for ILMLoss. "
                "Make sure that it is intentional."
            )

    def min_value(self, logits) -> float:
        if self._min_value is None:
            self._min_value = torch.finfo(logits.dtype).min
        return self._min_value

    def configure(self, pl_module: BaseLightningModule):
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def create_mask(
        self,
        non_drop_non_pad: Bool[TT, " batch seq_len"],
        constraint: Optional[Bool[TT, " batch seq_len"]],
    ) -> Bool[TT, " batch seq_len"]:
        if constraint is None or not self.use_constraint:
            return (non_drop_non_pad.logical_not_()[:, 1:]).unsqueeze(-1)
        elif self.use_constraint and constraint is not None:
            return (
                non_drop_non_pad.logical_not_().logical_or_(constraint)[:, 1:]
            ).unsqueeze(-1)
        else:
            raise ValueError("Invalid constraint")

    def get_compilable_functions(self) -> List[Callable]:
        return [self.loss_fn]

    def __call__(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMLossDict:
        sparse_target = batch["target_ids"]
        batch["target_ids"] = sparse_target.to_dense()
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        batch["target_ids"] = sparse_target
        return loss_dict

    def loss_fn(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMLossDict:
        x_0, token_type_ids, drop, target_ids, constraint = (
            batch["input_ids"],  # shape (batch, seq_len)
            batch["token_type_ids"],  # shape (batch, seq_len)
            batch["drop"],  # shape (batch, seq_len)
            batch["target_ids"],  # shape (batch, seq_len, vocab_size)
            batch["constraint"],  # shape (batch, seq_len)
        )
        # TODO (efficiency): This might be redundant if the data pipeline is aware of loss_on_padding flag.
        non_pad = (
            batch["attention_mask"].to(dtype=torch.bool)
            if not self.loss_on_padding
            else torch.ones_like(batch["attention_mask"], dtype=torch.bool)
        )  # shape (batch, seq_len)
        non_drop_non_pad = torch.logical_and(~drop, non_pad)
        drop = torch.logical_and(drop, non_pad)  # shape (batch, seq_len)
        positions = (
            torch.cumsum(non_drop_non_pad, dim=-1) - 1
        )  # shape (batch, seq_len)
        x_t = torch.where(drop, self.mask_token_id_tensor.view(1, 1), x_0)  # type: ignore
        model = cast(ILMModel, self.model)
        logits, _ = model(
            x_t,
            non_drop_non_pad,
            positions=positions,
            # token_type_ids=token_type_ids, model does not use it
            token_type_ids=constraint if self.input_constraint else None,
        )
        n_drops = drop.sum(dim=-1)
        p_target = target_ids / (n_drops[:, None, None] + 1e-9)
        loss_mask = self.create_mask(non_drop_non_pad, constraint)
        ce = masked_ce_last_two_dims(
            logits[:, 1:, :],
            p_target[:, 1:, :],
            loss_mask,
            min_value=self.min_value(logits),
            inplace=True,
        )  # shape (batch,)
        example_loss = ce
        loss = example_loss.mean()

        return {
            "loss": loss,
            "batch_loss": example_loss.detach(),
        }


class ILMLossWithMaskedCEForWrappedPaddedSequences(
    LossFunction[ILMBatch, ILMLossDict]
):
    def __init__(
        self,
        loss_on_padding: bool = False,
        model: Optional[ILMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_constraint: bool = False,
        expand_attention_mask: bool = True,
    ):
        self.loss_on_padding = loss_on_padding
        self.model = model
        self.tokenizer = tokenizer  # type: ignore
        self.use_constraint = use_constraint
        if use_constraint:
            raise ValueError(
                "use_constraint is not supported for ILMLossWithMaskedCEForWrappedPaddedSequences"
            )
        self._min_value: Optional[float] = None
        self.mask_token_id_tensor = None
        self.expand_attention_mask = expand_attention_mask
        if not self.loss_on_padding:
            logger.warning(
                "loss_on_padding is False for ILMLoss. "
                "Make sure that it is intentional."
            )

    @torch.no_grad()
    def expand_mask_create_positions_count_drops_cls_mask(
        self,
        token_type_ids: Bool[TT, " batch seq_len"],
        non_drop_non_pad: Bool[TT, " batch seq_len"],
        drop: Bool[TT, " batch seq_len"],
        batch_size: int,
        block_size: int,
    ) -> Tuple[
        Bool[TT, " batch seq_len seq_len"],
        Integer[TT, " batch seq_len"],
        Integer[TT, " batch seq_len"],
        Bool[TT, " batch seq_len"],
    ]:
        # mask
        # identify the positions of sequence boundaries
        sequence_boundary = token_type_ids == 0  # shape (batch, seq_len)
        # each subsequence group will get a unique integer id starting from 0
        subsequence_group = torch.cumsum(
            sequence_boundary, dim=-1
        )  # shape (batch, seq_len)
        # create a block diagonal mask with one block per subsequence group
        mask = (
            subsequence_group[:, :, None] == subsequence_group[:, None, :]
        ).logical_and_(
            non_drop_non_pad.unsqueeze(1)  # (batch, 1, seq_len)
        )  # shape (batch, seq_len, seq_len)

        # positions
        # create increasing sequence of positions
        arange_seq = (
            torch.arange(block_size, device=token_type_ids.device)
            .unsqueeze(0)
            .expand(batch_size, block_size)
        )  # shape (batch, seq_len), exapnd is view op
        # collect the position value at the start of each subsequence group and set the rest to 0
        # not the first subsequence group will start at position 0
        positions_at_start_of_group = torch.where(
            sequence_boundary,
            arange_seq,
            torch.zeros(
                (1,), dtype=torch.long, device=token_type_ids.device
            ).view(1, 1),
        )  # shape (batch, seq_len)
        # For each token get its group's start position
        cumulative_max, _ = torch.cummax(
            positions_at_start_of_group, dim=-1
        )  # ignore the indices, we want values
        # the position of each token is the global positions minus the group start position
        positions = arange_seq - cumulative_max
        # shape (batch, seq_len)

        # n_drops
        drop_int = drop.to(dtype=torch.long)
        n_drops_in_group = torch.zeros(
            (batch_size, block_size),
            dtype=torch.long,
            device=token_type_ids.device,
        )
        # for each token, add whether it is dropped (1) or not (0) into its group's count
        n_drops_in_group = n_drops_in_group.scatter_add_(
            dim=1, index=subsequence_group, src=drop_int
        )
        # Ex: n_drops_in_group[0]: [5, 3, 10, 0, 0, ....].
        # Here 5,3,10 is the number of tokens dropped in the first, second and third subsequence group respectively.
        # finaly assign these 5, 3, 10 to each token in their respective group
        n_drops = torch.gather(
            n_drops_in_group, dim=1, index=subsequence_group
        )
        return mask, positions, n_drops, sequence_boundary

    def min_value(self, logits) -> float:
        if self._min_value is None:
            self._min_value = torch.finfo(logits.dtype).min
        return self._min_value

    def configure(self, pl_module: BaseLightningModule):
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def create_mask(
        self,
        non_drop_non_pad: Bool[TT, " batch seq_len"],
        constraint: Optional[Bool[TT, " batch seq_len"]],
    ) -> Bool[TT, " batch seq_len"]:
        if constraint is None or not self.use_constraint:
            return (non_drop_non_pad.logical_not_()[:, 1:]).unsqueeze(-1)
        elif self.use_constraint and constraint is not None:
            return (
                non_drop_non_pad.logical_not_().logical_or_(constraint)[:, 1:]
            ).unsqueeze(-1)
        else:
            raise ValueError("Invalid constraint")

    def __call__(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMLossDict:
        sparse_target = batch["target_ids"]
        batch["target_ids"] = sparse_target.to_dense()
        # extract sizes here to make loss_fn compile friendly
        batch_size = batch["input_ids"].shape[0]
        block_size = batch["input_ids"].shape[1]
        loss_dict = self.loss_fn(
            batch,
            batch_size,
            block_size,
            batch_idx,
            dataloader_idx,
            dataloader_name,
        )
        batch["target_ids"] = sparse_target
        return loss_dict

    def loss_fn(
        self,
        batch: ILMBatch,
        batch_size: int,
        block_size: int,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMLossDict:
        x_0, token_type_ids, drop, target_ids = (
            batch["input_ids"],  # shape (batch, seq_len)
            batch["token_type_ids"],  # shape (batch, seq_len)
            batch["drop"],  # shape (batch, seq_len)
            batch["target_ids"],  # shape (batch, seq_len, vocab_size)
        )
        non_pad = (
            batch["attention_mask"].to(dtype=torch.bool)
            if not self.loss_on_padding
            else torch.ones_like(batch["attention_mask"], dtype=torch.bool)
        )  # shape (batch, seq_len)
        non_drop_non_pad = torch.logical_and(~drop, non_pad)
        # CLEANUP: remove the drop and non_pad check after ensuring that the datamodule handles it correctly
        # based on the loss_on_padding flag
        # drop = torch.logical_and(drop, non_pad)  # shape (batch, seq_len)
        assert (
            torch.logical_and(drop, ~non_pad).sum() == 0
            if self.loss_on_padding
            else True
        )
        x_t = torch.where(drop, self.mask_token_id_tensor.view(1, 1), x_0)  # type: ignore
        mask, positions, n_drops, cls = (
            self.expand_mask_create_positions_count_drops_cls_mask(
                token_type_ids, non_drop_non_pad, drop, batch_size, block_size
            )
        )
        cls[:, -1] = True
        model = cast(ILMModel, self.model)
        logits, _ = model(
            x_t,
            mask,
            positions=positions,
            # token_type_ids=token_type_ids, # typids are not used for constraints inside the model
        )
        # n_drops = n_drops.to(dtype=target_ids.dtype)
        p_target = target_ids / (n_drops.unsqueeze(-1) + 1e-9)
        ce = masked_ce_last_two_dims(
            logits,
            p_target,
            cls.unsqueeze(-1),
            min_value=self.min_value(logits),
            inplace=True,
        )  # shape (batch,)
        example_loss = ce
        loss = example_loss.mean()

        return {
            "loss": loss,
            "batch_loss": example_loss.detach(),
        }


class ILMWithLengthClassificationLossWithMaskedCE(
    LossFunction[ILMBatch, ILMWithLengthClassificationLossDict]
):
    def __init__(
        self,
        loss_on_padding: bool = False,
        model: Optional[ILMModelWithLengthClassification] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_constraint: bool = False,
        input_constraint: bool = False,
    ):
        self.loss_on_padding = loss_on_padding
        self.model = model
        self.tokenizer = tokenizer  # type: ignore
        self.mask_token_id_tensor = None
        self.use_constraint = use_constraint
        self.input_constraint = input_constraint
        self._min_value: Optional[float] = None
        if self.loss_on_padding:
            logger.warning(
                "loss_on_padding is True for ILMWithLengthClassificationLossWithMaskedCE. "
                "Make sure that it is intentional."
            )

    def min_value(self, logits) -> float:
        if self._min_value is None:
            self._min_value = torch.finfo(logits.dtype).min
        return self._min_value

    def create_mask(
        self,
        non_drop_non_pad: Bool[TT, " batch seq_len"],
        constraint: Optional[Bool[TT, " batch seq_len"]],
    ) -> Bool[TT, " batch seq_len 1"]:
        if constraint is None or not self.use_constraint:
            return (non_drop_non_pad.logical_not_()[:, 1:]).unsqueeze(-1)
        elif self.use_constraint and constraint is not None:
            return (
                non_drop_non_pad.logical_not_().logical_or_(constraint)[:, 1:]
            ).unsqueeze(-1)
        else:
            raise ValueError("Invalid constraint")

    def get_compilable_functions(self) -> List[Callable]:
        return [self.loss_fn]

    def configure(self, pl_module: BaseLightningModule):
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMWithLengthClassificationLossDict:
        sparse_target = batch["target_ids"]
        batch["target_ids"] = sparse_target.to_dense()
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        batch["target_ids"] = sparse_target
        return loss_dict

    def loss_fn(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMWithLengthClassificationLossDict:
        x_0, token_type_ids, drop, target_ids, constraint = (
            batch["input_ids"],  # shape (batch, seq_len)
            batch["token_type_ids"],  # shape (batch, seq_len)
            batch["drop"],  # shape (batch, seq_len)
            batch["target_ids"],  # shape (batch, seq_len, vocab_size)
            batch["constraint"],  # shape (batch, seq_len)
        )
        non_pad = (
            batch["attention_mask"].to(dtype=torch.bool)
            if not self.loss_on_padding
            else torch.ones_like(batch["attention_mask"], dtype=torch.bool)
        )  # shape (batch, seq_len)
        non_drop_non_pad = torch.logical_and(~drop, non_pad)
        drop = torch.logical_and(drop, non_pad)  # shape (batch, seq_len)
        positions = (
            torch.cumsum(non_drop_non_pad, dim=-1) - 1
        )  # shape (batch, seq_len)
        x_t = torch.where(drop, self.mask_token_id_tensor.view(1, 1), x_0)  # type: ignore
        model = cast(ILMModelWithLengthClassification, self.model)
        logits, length_logits = model(
            x_t,
            non_drop_non_pad,
            positions=positions,
            token_type_ids=constraint if self.input_constraint else None,
        )
        n_drops = drop.sum(dim=-1)
        p_target = target_ids.to_dense() / (n_drops[:, None, None] + 1e-9)
        loss_mask = self.create_mask(non_drop_non_pad, constraint)
        ce = masked_ce_last_two_dims(
            logits[:, 1:, :],
            p_target[:, 1:, :],
            loss_mask,
            min_value=self.min_value(logits),
            inplace=True,
        )  # shape (batch,)
        example_loss = ce
        length_loss = torch.nn.functional.cross_entropy(
            length_logits, n_drops, reduction="none"
        )  # shape (batch,)
        example_loss = ce + length_loss
        loss = example_loss.mean()

        return {
            "loss": loss,
            "batch_loss": example_loss.detach(),
            "per_example_length_loss": length_loss.detach(),
            "per_example_ce": ce.detach(),
            "length_logits": length_logits.detach(),
            "n_drops": n_drops.detach(),
        }


class ILMWithStoppingClassificationLossWithMaskedCE(
    LossFunction[ILMBatch, ILMWithStoppingClassificationLossDict]
):
    def __init__(
        self,
        loss_on_padding: bool = False,
        model: Optional[ILMModelWithStoppingClassification] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_constraint: bool = False,
        input_constraint: bool = False,
        stopping_class_weight: Optional[float] = None,
        stopping_loss_weight: Optional[float] = None,
    ):
        self.loss_on_padding = loss_on_padding
        self.model = model
        self.tokenizer = tokenizer  # type: ignore
        self.mask_token_id_tensor = None
        self.use_constraint = use_constraint
        self.input_constraint = input_constraint
        self._min_value: Optional[float] = None
        if self.loss_on_padding:
            logger.warning(
                "loss_on_padding is True for ILMWithStoppingClassificationLossWithMaskedCE. "
                "Make sure that it is intentional."
            )
        self.stopping_class_weight: Optional[torch.Tensor] = (
            stopping_class_weight
        )
        self.stopping_loss_weight: Optional[torch.Tensor] = (
            stopping_loss_weight
        )

    def min_value(self, logits) -> float:
        if self._min_value is None:
            self._min_value = torch.finfo(logits.dtype).min
        return self._min_value

    def configure(self, pl_module: BaseLightningModule):
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )
        if self.stopping_class_weight is not None:
            if not (0.0 <= self.stopping_class_weight <= 1.0):
                raise ValueError(
                    f"stopping_class_weight must be between 0 and 1, got {self.stopping_class_weight}"
                )
            self.stopping_class_weight = torch.tensor(
                [self.stopping_class_weight, 1.0 - self.stopping_class_weight],
                device=pl_module.device,
            )
        if self.stopping_loss_weight is not None:
            if not (0.0 <= self.stopping_loss_weight <= 1.0):
                raise ValueError(
                    f"stopping_loss_weight must be between 0 and 1, got {self.stopping_loss_weight}"
                )
            self.stopping_loss_weight = torch.tensor(
                float(self.stopping_loss_weight),
                device=pl_module.device,
            )
        else:
            self.stopping_loss_weight = torch.tensor(
                1.0, device=pl_module.device
            )

    def create_mask(
        self,
        non_drop_non_pad: Bool[TT, " batch seq_len"],
        constraint: Optional[Bool[TT, " batch seq_len"]],
    ) -> Bool[TT, " batch seq_len"]:
        if constraint is None or not self.use_constraint:
            return (non_drop_non_pad.logical_not_()[:, 1:]).unsqueeze(-1)
        elif self.use_constraint and constraint is not None:
            return (
                non_drop_non_pad.logical_not_().logical_or_(constraint)[:, 1:]
            ).unsqueeze(-1)
        else:
            raise ValueError("Invalid constraint")

    def get_compilable_functions(self) -> List[Callable]:
        return [self.loss_fn]

    def __call__(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMWithStoppingClassificationLossDict:
        sparse_target = batch["target_ids"]
        batch["target_ids"] = sparse_target.to_dense()
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        batch["target_ids"] = sparse_target
        return loss_dict

    def loss_fn(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMWithStoppingClassificationLossDict:
        x_0, token_type_ids, drop, target_ids, constraint = (
            batch["input_ids"],  # shape (batch, seq_len)
            batch["token_type_ids"],  # shape (batch, seq_len)
            batch["drop"],  # shape (batch, seq_len)
            batch["target_ids"],  # shape (batch, seq_len, vocab_size)
            batch["constraint"],  # shape (batch, seq_len)
        )
        if flags.DEBUG_CD_NAN:
            if x_0.isnan().any():
                indices = x_0.isnan().nonzero()
                raise ValueError(f"x_0 has nan at {indices}")

        non_pad = (
            batch["attention_mask"].to(dtype=torch.bool)
            if not self.loss_on_padding
            else torch.ones_like(batch["attention_mask"], dtype=torch.bool)
        )  # shape (batch, seq_len)
        non_drop_non_pad = torch.logical_and(~drop, non_pad)
        drop = torch.logical_and(drop, non_pad)  # shape (batch, seq_len)
        positions = (
            torch.cumsum(non_drop_non_pad, dim=-1) - 1
        )  # shape (batch, seq_len)
        x_t = torch.where(drop, self.mask_token_id_tensor.view(1, 1), x_0)  # type: ignore
        model = cast(ILMModelWithStoppingClassification, self.model)
        logits, stopping_logits = model(
            x_t,
            non_drop_non_pad,
            positions=positions,
            token_type_ids=constraint if self.input_constraint else None,
        )
        if flags.DEBUG_CD_NAN:
            message = ""
            if stopping_logits.isnan().any():
                indices = stopping_logits.isnan().nonzero()
                name, param = list(model.named_parameters())[0]
                if param.isnan().any():
                    message += f"{name} has nan\n"
                message += f"stopping_logits has nan at {indices}\n"
                raise ValueError(message)
            if logits.isnan().any():
                indices = logits.isnan().nonzero()
                name, param = list(model.named_parameters())[0]
                if param.isnan().any():
                    message += f"{name} has nan\n"
                message += f"logits has nan at {indices}\n"
                raise ValueError(message)

        n_drops = drop.sum(dim=-1)  # shape (batch,)
        p_target = target_ids.to_dense() / (
            n_drops[:, None, None] + 1e-9
        )  # shape (batch, seq_len, vocab_size)
        loss_mask = self.create_mask(non_drop_non_pad, constraint)
        ce = masked_ce_last_two_dims(
            logits[:, 1:, :],
            p_target[:, 1:, :],
            loss_mask,
            min_value=self.min_value(logits),
            inplace=True,
        )  # shape (batch,)
        example_loss = ce
        stopping_loss = (
            self.stopping_loss_weight
            * torch.nn.functional.cross_entropy(
                stopping_logits,
                (n_drops > 0).long(),
                weight=self.stopping_class_weight,
                reduction="none",
            )
        )  # shape (batch,) # class=0 is stopping, class=1 is not stopping
        if flags.DEBUG_CD_NAN:
            message = ""
            if stopping_loss.isnan().any():
                indices = stopping_loss.isnan().nonzero()
                name, param = list(model.named_parameters())[0]
                if param.isnan().any():
                    message += f"{name} has nan\n"
                message += f"stopping_loss has nan at {indices}\n"
                message += f"stopping_logits:\n{stopping_logits[indices]}\n"
            if stopping_loss.isinf().any():
                indices = stopping_loss.isinf().nonzero()
                message += f"stopping_loss has inf at {indices}\n"
                message += f"stopping_logits:\n{stopping_logits[indices]}\n"
            if example_loss.isnan().any():
                indices = example_loss.isnan().nonzero()
                name, param = list(model.named_parameters())[0]
                if param.isnan().any():
                    message += f"{name} has nan\n"
                message += f"example_loss has nan at {indices}\n"
            if example_loss.isinf().any():
                indices = example_loss.isinf().nonzero()
                message += f"example_loss has inf at {indices}\n"
            if message:
                raise ValueError(message)

        example_loss = ce + stopping_loss
        loss = example_loss.mean()

        return {
            "loss": loss,
            "batch_loss": example_loss.detach(),
            "per_example_stopping_loss": stopping_loss.detach(),
            "per_example_ce": ce.detach(),
            "stopping_logits": stopping_logits.detach(),
            "n_drops": n_drops.detach(),
        }


# endregion: Loss functions
################################################################

###############################################################
# region: Predictors


class ILMPredictorUtilitiesMixin:
    tokenizer: Tokenizer
    return_history: bool

    def clean_up_pred_ids(
        self, pred_ids: Integer[TT, " *batch seq_len"]
    ) -> Integer[TT, " *batch seq_len"]:
        """Remove mask tokens inserted due to batched prediction."""
        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = self.tokenizer.mask_token_id
        remove = torch.tensor(
            [mask_token_id, pad_token_id], device=pred_ids.device
        )
        clean_pred_ids = remove_tokens(pred_ids, remove, pad_token_id)
        return clean_pred_ids

    def decode(self, results: Dict) -> Tuple[
        List[str],
        List[str],
        Integer[TT, " batch seq_len"],
        Integer[TT, " batch seq_len"],
        Integer[TT, " batch seq_len"],
    ]:
        x: Integer[TT, " batch seq_len"] = results["x"]
        positions: Integer[TT, " batch seq_len"] = results["positions"]
        attention_mask: Bool[TT, " batch seq_len"] = results["attention_mask"]
        # all tensors are out of order. Sort them based on positions.
        final_positions, sorted_positions_indices = torch.sort(
            positions, dim=-1
        )
        final_x: Integer[TT, " batch seq_len"] = torch.gather(
            x, dim=-1, index=sorted_positions_indices
        )
        final_attention_mask: Bool[TT, " batch seq_len"] = torch.gather(
            attention_mask, dim=-1, index=sorted_positions_indices
        )
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            final_x, skip_special_tokens=False
        )
        out: List[str] = self.tokenizer.batch_decode(
            final_x, skip_special_tokens=True
        )
        return (
            out,
            out_with_spl_tokens,
            final_x,
            final_attention_mask,
            final_positions,
        )

    def _update_history(
        self,
        history: List[List[Tuple[str, float, int]]],
        step_results: Dict[str, Any],
        current_step: int,
    ) -> List[List[Tuple[str, float, int]]]:
        if not self.return_history:
            return history
        if (
            step_results["predict"] is not None
            and step_results["predict"].any().item()
        ):
            decoded_tuple = self.decode(step_results)
            for batch_idx in (
                step_results["predict"].nonzero().flatten().tolist()
            ):
                history[batch_idx].append(
                    (
                        decoded_tuple[0][batch_idx],
                        1.0,
                        current_step,
                    )
                )
        return history

    def to_dict(
        self,
        batch: ILMBatch,  # type: ignore
        preds: ILMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        preds_list: List[
            Tuple[str, str, List[int], List[Tuple[str, float, int]], float]
        ] = list(
            zip(
                preds["text"],
                preds["text_with_spl_tokens"],
                preds["ids"].tolist(),
                preds["history"],
                preds.get(
                    "time_taken", cycle([-1])
                ),  # -1 when the predict method does not measure time.
            )
        )
        dicts: List[Dict[str, Any]] = []
        for text, text_with_spl_tokens, ids, history, time_taken in preds_list:
            rounded_history = [
                [subseq, round(t, 4), step] for subseq, t, step in history
            ]
            dicts.append(
                {
                    "text": text,
                    "text_with_spl_tokens": text_with_spl_tokens,
                    "ids": ids,
                    "history": rounded_history,
                    "time_taken": time_taken,
                }
            )
        return dicts


class ILMPredictor(
    torch.nn.Module,
    ILMPredictorUtilitiesMixin,
    Predictor[ILMBatch, ILMPredictionDict],
):
    token_ids_to_suppress: Integer[TT, " n_tokens_to_suppress"]

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        tokens_to_suppress: Optional[List[str]] = None,
        return_history: bool = False,
        sampling_method: Literal[
            "sample", "sample_top_k", "sample_top_p"
        ] = "sample",
        top: int = 1000,
        p: float = 0.9,
        second_sampling_method: Optional[
            Literal["sample", "sample_top_k", "sample_top_p"]
        ] = None,
        second_top: int = 1000,
        second_p: float = 0.9,
        model: Optional[ILMModel] = None,
        input_constraint: bool = False,
    ):
        """Constructor for ILMPredictor.

        Args:
            max_steps (int): The maximum number of steps to take.
            max_length (int): The maximum length (excluding special tokens like PAD and MASK)
                of the generated text.
            stopping_threshold (float): The threshold for stopping use on the length classification scores.
            tokenizer (Tokenizer): The tokenizer. Typically, set after initialization but before calling predict.
            noise_schedule (NoiseSchedule): The noise schedule. Typically, set after initialization but before calling predict.
            tokens_to_suppress (List[str]): The tokens to suppress during generation.
            return_history (bool): Whether to return the history.
            sampling_method (Literal["sample", "sample_top_k", "sample_top_p"]): The sampling method.
                When `second_sampling_method` is not provided, the specified method here is
                used to sample from the joint distribution of positions and tokens.
                When `second_sampling_method` is provided, the specified method here is
                used to sample from the token distribution (conditional) given the postions sampled
                using the `second_sampling_method`.
                "sample" means vanilla sampling from the distribution.
                "sample_top_k" means sampling from the top-k distribution.
                "sample_top_p" means sampling from the top-p distribution (neuclius samplingn).
            top (int): The top-k sampling parameter for `sampling_method`.
            p (float): The top-p sampling parameter for `sampling_method`.
            second_sampling_method (Optional[Literal["sample", "sample_top_k", "sample_top_p"]]): The second sampling method.
            second_top (int): The second top-k sampling parameter for `second_sampling_method`.
            second_p (float): The second top-p sampling parameter for `second_sampling_method`.
            model (Optional[ILMModel]): The model. Typically, set after initialization but before calling predict.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        token_ids_to_suppress = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(token)
                for token in (
                    tokens_to_suppress
                    or [
                        tokenizer.mask_token,
                        tokenizer.eos_token,
                        # tokenizer.pad_token, # don't suppress pad tokens
                        tokenizer.cls_token,
                        tokenizer.bos_token,
                    ]
                )
            ],
            dtype=torch.long,
            requires_grad=False,
        )
        super().__init__()
        self.model = model
        self.register_buffer(
            "token_ids_to_suppress", token_ids_to_suppress, persistent=False
        )
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        if sampling_method == "sample":
            self.sampling_function = sample_from_logits
        elif sampling_method == "sample_top_k":
            self.sampling_function = partial(sample_from_top_k, top)
        elif sampling_method == "sample_top_p":
            self.sampling_function = partial(sample_from_top_p, p)
        else:
            raise ValueError(f"Invalid sampling method: {sampling_method}")

        if second_sampling_method == "sample":
            self.second_sampling_function = sample_from_logits
        elif second_sampling_method == "sample_top_k":
            self.second_sampling_function = partial(
                sample_from_top_k, second_top
            )
        elif second_sampling_method == "sample_top_p":
            self.second_sampling_function = partial(
                sample_from_top_p, second_p
            )
        elif second_sampling_method is None:
            self.second_sampling_function = None
        else:
            raise ValueError(
                f"Invalid second sampling method: {second_sampling_method}"
            )

        self.noise_schedule = noise_schedule
        self.return_history = return_history
        self.input_constraint = input_constraint

    def _predict_single_step(
        self,
        step_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        TODO (doc): Add docstring.
        Constraints:
            - Mask tokens cannot be predicted
            - Input non-mask tokens cannot be changed
        """
        # fmt: off
        x_t: Integer[TT, " *batch seq_len"] = step_results["x"]
        positions: Integer[TT, " *batch seq_len"] = step_results["positions"]
        attention_mask: Bool[TT, " *batch seq_len"] = step_results["attention_mask"]
        constraint: Bool[TT, " *batch seq_len"] = step_results["constraint"]
        token_type_ids: Integer[TT, " *batch seq_len"] = step_results["token_type_ids"]
        # fmt: on
        model = cast(ILMModel, self.model)
        logits, _ = model(
            x_t,
            attention_mask,
            positions=positions,
            token_type_ids=constraint if self.input_constraint else None,
        )

        # suppress some specified (mostly special) tokens
        logits[:, :, self.token_ids_to_suppress] = -torch.inf
        # suppress predictions from input tokens that are mask or pad or part of the prefix
        suppress_positions = torch.logical_or(~attention_mask, constraint)
        logits = torch.where(
            suppress_positions.unsqueeze(-1),
            -torch.inf,
            logits,
        )
        # make sure that predict never goes from False to True
        pred_seq_index, pred_vocab_index = general_sample_over_last_two_dims(
            logits, self.sampling_function, self.second_sampling_function
        )  # shape (batch,), (batch,)
        # pred_seq_index is not the real index in the token sequence because logits and x_t are kept out of order.
        pred_real_index = positions.gather(
            dim=-1, index=pred_seq_index.unsqueeze(-1)
        ).squeeze(
            -1
        )  # shape (batch,)
        inserted_tokens = pred_vocab_index
        # predict = step_results["predict"]
        current_length = attention_mask.sum(dim=-1)
        predict = current_length < self.max_length
        x_s = torch.cat(
            [x_t, inserted_tokens.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        pred_attention_mask = torch.cat(
            [attention_mask, predict.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        pos_greater_than_inserted_position = (
            positions > pred_real_index.unsqueeze(-1)
        )  # mask of shape (batch, current_seq_len)
        pos_greater_than_inserted_position = torch.logical_and(
            pos_greater_than_inserted_position,
            predict.unsqueeze(-1),
        )
        inserted_positions = pred_real_index + 1
        # increment positions greater than inserted position
        pred_positions = positions + pos_greater_than_inserted_position.to(
            dtype=positions.dtype
        )
        pred_positions = torch.cat(
            [pred_positions, inserted_positions.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        constraint = torch.cat(
            [
                step_results["constraint"],
                torch.zeros(
                    (pred_positions.shape[0], 1),
                    device=step_results["constraint"].device,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )
        token_type_ids = torch.cat(
            [
                token_type_ids,
                torch.full(
                    (token_type_ids.shape[0], 1),
                    2,  # non-prefix token type id
                    device=token_type_ids.device,
                    dtype=token_type_ids.dtype,
                ),
            ],
            dim=-1,
        )

        step_result = {
            "x": x_s,
            "positions": pred_positions,
            "attention_mask": pred_attention_mask,
            "token_type_ids": token_type_ids,
            "constraint": constraint,
            "predict": predict,
        }
        return step_result

    def _stop(
        self,
        step_results: Dict[str, Any],
        current_step: int,
    ) -> bool:

        max_steps_reached = current_step >= self.max_steps
        return max_steps_reached

    @torch._dynamo.disable()
    def predict(
        self,
        batch: ILMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMPredictionDict:
        _start_time = time.time()
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        positions = attention_mask.cumsum(dim=-1) - 1
        if batch["constraint"] is not None:
            constraint = batch["constraint"]
        else:
            # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
            constraint = batch["token_type_ids"] == 0
        current_length = attention_mask.sum(dim=-1)
        predict = current_length < self.max_length
        step_results = {
            "x": batch["input_ids"],
            "positions": positions,
            "attention_mask": attention_mask,
            "token_type_ids": batch["token_type_ids"],
            "length_logits": None,
            "constraint": constraint,
            "oracle_length": batch.get("oracle_length", None),
            "predict": predict,
        }
        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)
        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(
                step_results,
            )
            history = self._update_history(history, step_results, current_step)
            current_step += 1
        # final step (nothing special)
        step_results = self._predict_single_step(
            step_results,
        )
        history = self._update_history(history, step_results, current_step)
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
            final_attention_mask,
            final_positions,
        ) = self.decode(step_results)
        _end_time = time.time()
        _time_taken = _end_time - _start_time
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "attention_mask": final_attention_mask,
            "positions": final_positions,
            "history": history,
            "loss": None,
            "time_taken": (
                [_time_taken] * len(out)
            ),  # cannot separate time for each sample
        }


class ILMPredictorWithLengthClassification(
    torch.nn.Module,
    ILMPredictorUtilitiesMixin,
    Predictor[ILMBatch, ILMPredictionDict],
):
    token_ids_to_suppress: Integer[TT, " n_tokens_to_suppress"]

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        stopping_threshold: float = 0.5,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        tokens_to_suppress: Optional[List[str]] = None,
        return_history: bool = False,
        sampling_method: Literal[
            "sample", "sample_top_k", "sample_top_p"
        ] = "sample",
        top: int = 1000,
        p: float = 0.9,
        second_sampling_method: Optional[
            Literal["sample", "sample_top_k", "sample_top_p"]
        ] = None,
        second_top: int = 1000,
        second_p: float = 0.9,
        model: Optional[ILMModel] = None,
        force_predict_first_step: bool = False,
        input_constraint: bool = False,
    ):
        """Constructor for ILMPredictor.

        Args:
            max_steps (int): The maximum number of steps to take.
            max_length (int): The maximum length (excluding special tokens like PAD and MASK)
                of the generated text.
            stopping_threshold (float): The threshold for stopping use on the length classification scores.
            tokenizer (Tokenizer): The tokenizer. Typically, set after initialization but before calling predict.
            noise_schedule (NoiseSchedule): The noise schedule. Typically, set after initialization but before calling predict.
            tokens_to_suppress (List[str]): The tokens to suppress during generation.
            return_history (bool): Whether to return the history.
            sampling_method (Literal["sample", "sample_top_k", "sample_top_p"]): The sampling method.
                When `second_sampling_method` is not provided, the specified method here is
                used to sample from the joint distribution of positions and tokens.
                When `second_sampling_method` is provided, the specified method here is
                used to sample from the token distribution (conditional) given the postions sampled
                using the `second_sampling_method`.
                "sample" means vanilla sampling from the distribution.
                "sample_top_k" means sampling from the top-k distribution.
                "sample_top_p" means sampling from the top-p distribution (neuclius samplingn).
            top (int): The top-k sampling parameter for `sampling_method`.
            p (float): The top-p sampling parameter for `sampling_method`.
            second_sampling_method (Optional[Literal["sample", "sample_top_k", "sample_top_p"]]): The second sampling method.
            second_top (int): The second top-k sampling parameter for `second_sampling_method`.
            second_p (float): The second top-p sampling parameter for `second_sampling_method`.
            model (Optional[ILMModel]): The model. Typically, set after initialization but before calling predict.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        token_ids_to_suppress = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(token)
                for token in (
                    tokens_to_suppress
                    or [
                        tokenizer.mask_token,
                        tokenizer.eos_token,
                        tokenizer.pad_token,
                        tokenizer.cls_token,
                        tokenizer.bos_token,
                    ]
                )
            ],
            dtype=torch.long,
            requires_grad=False,
        )
        super().__init__()
        self.stopping_threshold = stopping_threshold
        self.model = model
        self.register_buffer(
            "token_ids_to_suppress", token_ids_to_suppress, persistent=False
        )
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        if sampling_method == "sample":
            self.sampling_function = sample_from_logits
        elif sampling_method == "sample_top_k":
            self.sampling_function = partial(sample_from_top_k, top)
        elif sampling_method == "sample_top_p":
            self.sampling_function = partial(sample_from_top_p, p)
        else:
            raise ValueError(f"Invalid sampling method: {sampling_method}")

        if second_sampling_method == "sample":
            self.second_sampling_function = sample_from_logits
        elif second_sampling_method == "sample_top_k":
            self.second_sampling_function = partial(
                sample_from_top_k, second_top
            )
        elif second_sampling_method == "sample_top_p":
            self.second_sampling_function = partial(
                sample_from_top_p, second_p
            )
        elif second_sampling_method is None:
            self.second_sampling_function = None
        else:
            raise ValueError(
                f"Invalid second sampling method: {second_sampling_method}"
            )

        self.noise_schedule = noise_schedule
        self.return_history = return_history
        self.force_predict_first_step = force_predict_first_step
        self.input_constraint = input_constraint

    def _compute_stopping_mask(
        self,
        step_results: Dict[str, Any],
    ) -> Bool[TT, " batch"]:
        length_logits = step_results["length_logits"]
        if length_logits is not None:  # graph break?
            attention_mask = step_results["attention_mask"]
            p = torch.softmax(length_logits, dim=-1)[:, 0]
            predict = p < self.stopping_threshold
            predict.logical_and_(attention_mask.sum(-1) < self.max_length)
        else:
            predict = step_results["predict"]
        oracle_length = step_results.get("oracle_length", None)
        if oracle_length is not None:
            # don't do this update in-place
            predict = predict.logical_and(
                attention_mask.sum(-1) < oracle_length
            )
        return predict

    def _predict_single_step(
        self,
        step_results: Dict[str, Any],
        current_step: int,
    ) -> Dict[str, Any]:
        """
        TODO (doc): Add docstring.
        Constraints:
            - Mask tokens cannot be predicted
            - Input non-mask tokens cannot be changed
        """
        # fmt: off
        x_t: Integer[TT, " *batch seq_len"] = step_results["x"]
        positions: Integer[TT, " *batch seq_len"] = step_results["positions"]
        attention_mask: Bool[TT, " *batch seq_len"] = step_results["attention_mask"]
        constraint: Bool[TT, " *batch seq_len"] = step_results["constraint"]
        token_type_ids: Integer[TT, " *batch seq_len"] = step_results["token_type_ids"]
        # fmt: on
        model = cast(ILMModelWithLengthClassification, self.model)
        logits, length_logits = model(
            x_t,
            attention_mask,
            positions=positions,
            token_type_ids=constraint if self.input_constraint else None,
        )

        # suppress some specified (mostly special) tokens
        logits[:, :, self.token_ids_to_suppress] = -torch.inf
        # suppress predictions from input tokens that are mask or pad or part of the prefix
        suppress_positions = torch.logical_or(~attention_mask, constraint)
        logits = torch.where(
            suppress_positions.unsqueeze(-1),
            -torch.inf,
            logits,
        )
        step_results["length_logits"] = length_logits
        predict = self._compute_stopping_mask(step_results)
        if current_step == 1 and self.force_predict_first_step:
            predict = torch.ones_like(predict)
        # make sure that predict never goes from False to True
        if flags.DEBUG_OVERFIT:
            if step_results["predict"] is not None:
                assert not bool(
                    torch.logical_and(
                        ~step_results["predict"],
                        predict,
                    ).any()
                ), "predict went from False to True at indices {}".format(
                    (
                        torch.logical_and(
                            ~step_results["predict"],
                            predict,
                        ).nonzero()
                    )
                )
        else:
            # if predict was false, it should still be false
            predict = predict.logical_and(step_results["predict"])
        if not predict.any().item():
            return {
                "x": x_t,
                "positions": positions,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "length_logits": length_logits,
                "constraint": step_results["constraint"],
                "oracle_length": step_results.get("oracle_length", None),
                "predict": predict,
            }
        pred_seq_index, pred_vocab_index = general_sample_over_last_two_dims(
            logits, self.sampling_function, self.second_sampling_function
        )  # shape (batch,), (batch,)
        # pred_seq_index is not the real index in the token sequence because logits and x_t are kept out of order.
        pred_real_index = positions.gather(
            dim=-1, index=pred_seq_index.unsqueeze(-1)
        ).squeeze(
            -1
        )  # shape (batch,)
        inserted_tokens = torch.where(
            predict,
            pred_vocab_index,
            self.tokenizer.mask_token_id,
        )
        x_s = torch.cat(
            [x_t, inserted_tokens.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        pred_attention_mask = torch.cat(
            [attention_mask, predict.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        pos_greater_than_inserted_position = (
            positions > pred_real_index.unsqueeze(-1)
        )  # mask of shape (batch, current_seq_len)
        pos_greater_than_inserted_position = torch.logical_and(
            pos_greater_than_inserted_position,
            predict.unsqueeze(-1),
        )
        inserted_positions = pred_real_index + 1
        # increment positions greater than inserted position
        pred_positions = positions + pos_greater_than_inserted_position.to(
            dtype=positions.dtype
        )
        pred_positions = torch.cat(
            [pred_positions, inserted_positions.unsqueeze(-1)], dim=-1
        )  # shape (*batch, current_seq_len + 1)
        constraint = torch.cat(
            [
                step_results["constraint"],
                torch.zeros(
                    (pred_positions.shape[0], 1),
                    device=step_results["constraint"].device,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )
        token_type_ids = torch.cat(
            [
                token_type_ids,
                torch.full(
                    (token_type_ids.shape[0], 1),
                    2,  # non-prefix token type id
                    device=token_type_ids.device,
                    dtype=token_type_ids.dtype,
                ),
            ],
            dim=-1,
        )

        step_result = {
            "x": x_s,
            "positions": pred_positions,
            "attention_mask": pred_attention_mask,
            "token_type_ids": token_type_ids,
            "length_logits": length_logits,
            "constraint": constraint,
            "oracle_length": step_results.get("oracle_length", None),
            "predict": predict,
        }
        return step_result

    def _stop(
        self,
        step_results: Dict[str, Any],
        current_step: int,
    ) -> bool:
        max_steps_reached = current_step >= self.max_steps
        batch_done = not bool(step_results["predict"].any().item())
        return max_steps_reached or batch_done

    @torch._dynamo.disable()
    def predict(
        self,
        batch: ILMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ILMPredictionDict:
        _start_time = time.time()
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        positions = attention_mask.cumsum(dim=-1) - 1
        if batch["constraint"] is not None:
            constraint = batch["constraint"]
        else:
            # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
            constraint = batch["token_type_ids"] == 0

        step_results = {
            "x": batch["input_ids"],
            "positions": positions,
            "attention_mask": attention_mask,
            "token_type_ids": batch["token_type_ids"],
            "length_logits": None,
            "constraint": constraint,
            "oracle_length": batch.get("oracle_length", None),
            "predict": torch.ones(
                (batch["input_ids"].shape[0],),
                dtype=torch.bool,
                device=batch["input_ids"].device,
            ),
        }
        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)
        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(
                step_results,
                current_step,
            )
            history = self._update_history(history, step_results, current_step)
            current_step += 1
        # final step (nothing special)
        step_results = self._predict_single_step(
            step_results,
            current_step,
        )
        history = self._update_history(history, step_results, current_step)
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
            final_attention_mask,
            final_positions,
        ) = self.decode(step_results)
        _end_time = time.time()
        _time_taken = _end_time - _start_time
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "attention_mask": final_attention_mask,
            "positions": final_positions,
            "history": history,
            "loss": None,
            "time_taken": [_time_taken] * len(out),
        }


class ILMPredictorWithStoppingClassification(
    ILMPredictorWithLengthClassification
):
    def __init__(
        self,
        *args,
        stopping_threshold: float = 0.9,
        **kwargs,
    ):
        super().__init__(
            *args, stopping_threshold=stopping_threshold, **kwargs
        )


# endregion: Predictors
###############################################################


class ILMLightningModule(BaseLightningModule):
    predictor: Union[
        ILMPredictor,
        ILMPredictorWithLengthClassification,
        ILMPredictorWithStoppingClassification,
    ]

    def setup_predictor(self):
        self.predictor = hydra.utils.instantiate(
            self.config.predictor,
            tokenizer=self.tokenizer,
            noise_schedule=self.noise_schedule,
        )

    def setup_metrics(self):

        # Accumulated loss
        self.train_accumulated_loss = MeanMetric()
        self.val_accumulated_loss = MeanMetric()
        self.test_accumulated_loss = MeanMetric()

    def _prepare_input_batch_for_predict(self, batch: ILMBatch) -> ILMBatch:
        """Use the `drop` tensor in batch to update `attention_mask` and `input_ids`.

        drop and pad positions are changed to 0 and the rest are set to 1 in `attention_mask`.
        `input_ids` is set to the mask token for all drop positions. Use this function on a batch
        if the batch contains ground truth ids but the `drop` is set.

        Note:
            This function will do nothing if the `drop` is not set or if the `drop` tensor is all zeros.
        """
        # clone the batch
        cloned_batch: ILMBatch = {}  # type: ignore
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                cloned_batch[k] = v.clone()
            else:
                cloned_batch[k] = v
        non_pad_and_non_drop = torch.logical_and(
            batch["attention_mask"] == 1, batch["drop"] == 0
        )
        cloned_batch["attention_mask"] = non_pad_and_non_drop
        cloned_batch["input_ids"][batch["drop"]] = self.tokenizer.mask_token_id
        return cloned_batch

    def compute_loss(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Union[Dict[str, Any], ILMPredictionDict]:
        """
        Computes loss based on the dataloader name.

        For 'lm', the loss function is applied.
        For 'prediction', the predictor's predict_step is used.
        """
        if dataloader_name == "lm":
            return self.loss_function(
                batch, batch_idx, dataloader_idx, dataloader_name
            )
        elif dataloader_name == "prediction":
            cloned_batch = self._prepare_input_batch_for_predict(batch)
            return self.predictor.predict(
                cloned_batch, batch_idx, dataloader_idx, dataloader_name
            )
        else:
            return {}

    def update_train_metrics(
        self,
        batch: ILMBatch,
        loss_dict: ILMLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update training metrics with values computed by the loss function.
        """
        if dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")
        self.train_accumulated_loss.update(loss_dict["batch_loss"])

    def log_train_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log training metrics.
        """
        self.log(
            "train/loss",
            metrics["loss"].detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
            rank_zero_only=True,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log_dict(
            {
                "train/accumulated_loss": self.train_accumulated_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        # Optionally, remove non-essential elements from metrics
        for key in [
            "batch_loss",
        ]:
            metrics.pop(key, None)

    def update_val_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update validation metrics.
        """
        if dataloader_name == "lm":
            self.val_accumulated_loss.update(loss_dict["batch_loss"])
        elif dataloader_name == "prediction":
            pass  # For prediction branch, no metric update is performed for base IDLM

    def log_val_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log validation metrics.
        """
        if dataloader_name == "lm":
            self.log_dict(
                {
                    "val/accumulated_loss": self.val_accumulated_loss,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )
            for key in [
                "batch_loss",
            ]:
                metrics.pop(key, None)
        elif dataloader_name == "prediction":
            # Write to file and logger.
            self.log_predictions(batch, metrics, "val", dataloader_name)

    def update_test_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update test metrics.
        """
        if dataloader_name == "lm":
            self.test_accumulated_loss.update(loss_dict["batch_loss"])
        elif dataloader_name == "prediction":
            pass  # For prediction branch, no metric update is performed for base IDLM
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_test_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log test metrics.
        """
        if dataloader_name == "lm":
            self.log_dict(
                {
                    "test/accumulated_loss": self.test_accumulated_loss,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )
            for key in [
                "batch_loss",
            ]:
                metrics.pop(key, None)
        elif dataloader_name == "prediction":
            # Write to file and logger.
            self.log_predictions(batch, metrics, "test", f"{dataloader_name}")
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def update_predict_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update metrics during prediction (if needed).
        Currently, no extra metrics are tracked for prediction.
        """
        pass

    def log_predict_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log prediction metrics (if needed).
        """
        pass

    def reset_train_metrics(self) -> None:
        self.train_accumulated_loss.reset()

    def reset_val_metrics(self) -> None:
        self.val_accumulated_loss.reset()

    def reset_test_metrics(self) -> None:
        self.test_accumulated_loss.reset()


class ILMWithLengthClassificationLightningModule(ILMLightningModule):
    predictor: Union[
        ILMPredictor,
        ILMPredictorWithLengthClassification,
        ILMPredictorWithStoppingClassification,
    ]
    length_logits_key: str = "length_logits"
    length_loss_key: str = "per_example_length_loss"

    def setup_metrics(self):
        super().setup_metrics()
        # Initialize diagnostic metrics
        self.train_ce_metric = MeanMetric()
        self.val_ce_metric = MeanMetric()
        self.test_ce_metric = MeanMetric()
        # length related metrics
        self.train_length_loss_metric = MeanMetric()
        self.val_length_loss_metric = MeanMetric()
        self.test_length_loss_metric = MeanMetric()
        # try to use the threshold from the predictor
        threshold = 0.7
        if self.predictor is not None:
            if hasattr(self.predictor, "stopping_threshold"):
                threshold = self.predictor.stopping_threshold

        self.train_stopping_accuracy = BinaryAccuracy(threshold=threshold)
        self.val_stopping_accuracy = BinaryAccuracy(threshold=threshold)
        self.test_stopping_accuracy = BinaryAccuracy(threshold=threshold)
        # TODO: Implement AUPRC to make sure that we will be able to find a threshold after training
        # https://github.com/Lightning-AI/torchmetrics/discussions/698

    def _prepare_input_batch_for_predict(self, batch: ILMBatch) -> ILMBatch:
        """Use the `drop` tensor in batch to update `attention_mask` and `input_ids`.

        drop and pad positions are changed to 0 and the rest are set to 1 in `attention_mask`.
        `input_ids` is set to the mask token for all drop positions. Use this function on a batch
        if the batch contains ground truth ids but the `drop` is set.

        Note:
            This function will do nothing if the `drop` is not set or if the `drop` tensor is all zeros.
        """
        # clone the batch
        cloned_batch: IDLMBatch = {}  # type: ignore
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                cloned_batch[k] = v.clone()
            else:
                cloned_batch[k] = v
        non_pad_and_non_drop = torch.logical_and(
            batch["attention_mask"] == 1, batch["drop"] == 0
        )
        cloned_batch["attention_mask"] = non_pad_and_non_drop
        cloned_batch["input_ids"][batch["drop"]] = self.tokenizer.mask_token_id
        return cloned_batch

    def compute_loss(
        self,
        batch: ILMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Union[ILMWithLengthClassificationLossDict, ILMPredictionDict, None]:
        """
        Computes loss based on the dataloader name.

        For 'lm', the loss function is applied.
        For 'prediction', the predictor's predict_step is used.
        """
        if dataloader_name == "lm":
            return self.loss_function(
                batch, batch_idx, dataloader_idx, dataloader_name
            )
        elif dataloader_name == "prediction":
            cloned_batch = self._prepare_input_batch_for_predict(batch)
            return self.predictor.predict(
                cloned_batch, batch_idx, dataloader_idx, dataloader_name
            )
        else:
            return None

    def update_train_metrics(
        self,
        batch: ILMBatch,
        loss_dict: ILMWithLengthClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update training metrics with values computed by the loss function.
        """
        super().update_train_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")
        self.train_ce_metric.update(loss_dict["per_example_ce"])
        self.train_length_loss_metric.update(loss_dict[self.length_loss_key])
        length_zero_p = torch.softmax(
            loss_dict[self.length_logits_key], dim=-1
        )[:, 0]
        self.train_stopping_accuracy.update(
            length_zero_p,
            (loss_dict["n_drops"] == 0).to(torch.long),
        )

    def log_train_metrics(
        self,
        batch: ILMBatch,
        metrics: ILMWithLengthClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log training metrics.
        """
        super().log_train_metrics(batch, metrics, batch_idx, dataloader_idx, dataloader_name)  # type: ignore
        self.log_dict(
            {
                "train/ce": self.train_ce_metric,
                "train/length_loss": self.train_length_loss_metric,
                "train/stopping_accuracy": self.train_stopping_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        # Optionally, remove non-essential elements from metrics
        for key in [
            "per_example_ce",
            self.length_loss_key,
            self.length_logits_key,
            "n_drops",
        ]:
            metrics.pop(key, None)  # type: ignore

    def update_val_metrics(
        self,
        batch: ILMBatch,
        loss_dict: Union[
            ILMWithLengthClassificationLossDict, ILMPredictionDict
        ],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update validation metrics.
        """
        super().update_val_metrics(batch, loss_dict, batch_idx, dataloader_idx, dataloader_name)  # type: ignore
        if dataloader_name == "lm":
            loss_dict = cast(ILMWithLengthClassificationLossDict, loss_dict)
            self.val_ce_metric.update(loss_dict["per_example_ce"])
            self.val_length_loss_metric.update(loss_dict[self.length_loss_key])
            length_zero_p = torch.softmax(
                loss_dict[self.length_logits_key], dim=-1
            )[:, 0]
            self.val_stopping_accuracy.update(
                length_zero_p,
                (loss_dict["n_drops"] == 0).to(torch.long),
            )
            for key in [
                "per_example_ce",
                self.length_loss_key,
                self.length_logits_key,
                "n_drops",
            ]:
                loss_dict.pop(key, None)  # type: ignore
        elif dataloader_name == "prediction":
            pass  # For prediction branch, no metric update is performed for base IDLM

    def log_val_metrics(
        self,
        batch: ILMBatch,
        metrics: Union[ILMWithLengthClassificationLossDict, ILMPredictionDict],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log validation metrics.
        """
        super().log_val_metrics(batch, metrics, batch_idx, dataloader_idx, dataloader_name)  # type: ignore
        if dataloader_name == "lm":
            metrics = cast(ILMWithLengthClassificationLossDict, metrics)
            self.log_dict(
                {
                    "val/ce": self.val_ce_metric,
                    "val/length_loss": self.val_length_loss_metric,
                    "val/stopping_accuracy": self.val_stopping_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )
            for key in [
                "per_example_ce",
                self.length_loss_key,
                self.length_logits_key,
                "n_drops",
            ]:
                metrics.pop(key, None)  # type: ignore

    def update_test_metrics(
        self,
        batch: ILMBatch,
        loss_dict: Union[
            ILMWithLengthClassificationLossDict, ILMPredictionDict
        ],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update test metrics.
        """
        super().update_test_metrics(batch, loss_dict, batch_idx, dataloader_idx, dataloader_name)  # type: ignore
        if dataloader_name == "lm":
            loss_dict = cast(ILMWithLengthClassificationLossDict, loss_dict)
            self.test_ce_metric.update(loss_dict["per_example_ce"])
            self.test_length_loss_metric.update(
                loss_dict[self.length_loss_key]
            )
            length_zero_p = torch.softmax(
                loss_dict[self.length_logits_key], dim=-1
            )[:, 0]
            self.test_stopping_accuracy.update(
                length_zero_p,
                (loss_dict["n_drops"] == 0).to(torch.long),
            )
            for key in [
                "per_example_ce",
                self.length_loss_key,
                self.length_logits_key,
                "n_drops",
            ]:
                loss_dict.pop(key, None)  # type: ignore
        elif dataloader_name == "prediction":
            pass  # For prediction branch, no metric update is performed for base IDLM
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_test_metrics(
        self,
        batch: ILMBatch,
        metrics: Union[ILMWithLengthClassificationLossDict, ILMPredictionDict],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log test metrics.
        """
        if dataloader_name == "lm":
            self.log_dict(
                {
                    "test/ce": self.test_ce_metric,
                    "test/length_loss": self.test_length_loss_metric,
                    "test/stopping_accuracy": self.test_stopping_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
            for key in [
                "per_example_ce",
                self.length_loss_key,
                self.length_logits_key,
                "n_drops",
            ]:
                metrics.pop(key, None)  # type: ignore
        elif dataloader_name == "prediction":
            pass  # For prediction branch, no metric update is performed for base IDLM
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def reset_train_metrics(self) -> None:
        super().reset_train_metrics()
        self.train_ce_metric.reset()
        self.train_length_loss_metric.reset()
        self.train_stopping_accuracy.reset()

    def reset_val_metrics(self) -> None:
        super().reset_val_metrics()
        self.val_ce_metric.reset()
        self.val_length_loss_metric.reset()
        self.val_stopping_accuracy.reset()

    def reset_test_metrics(self) -> None:
        super().reset_test_metrics()
        self.test_ce_metric.reset()
        self.test_length_loss_metric.reset()
        self.test_stopping_accuracy.reset()


class ILMWithStoppingClassificationLightningModule(
    ILMWithLengthClassificationLightningModule
):
    length_logits_key: str = "stopping_logits"
    length_loss_key: str = "per_example_stopping_loss"


# endregion: Base ILM
################################################################################


################################################################################
# region: IDLM for Star Graphs


#######################################################################
# region: Specialized metrics for Star Graphs


class ExactMatch(MeanMetric):
    def update(
        self,
        pred: Integer[TT, " *batch seq_len"],
        target: Integer[TT, " *batch seq_len"],
        pred_length: Optional[Integer[TT, " *batch"]] = None,
        target_length: Optional[Integer[TT, " *batch"]] = None,
    ):
        """
        Args:
            pred: predicted tokens
            target: target tokens
            pred_length: length of the predicted tokens
            target_length: length of the target tokens
        """
        matches = (
            (pred == target).all(dim=-1).to(torch.float32)
        )  # shape (*batch)
        if pred_length is not None and target_length is not None:
            matches = matches * (pred_length == target_length)
        super().update(matches)


class HammingAccuracy(MeanMetric):
    def update(
        self,
        pred: Integer[TT, " *batch seq_len"],
        target: Integer[TT, " *batch seq_len"],
        pred_mask: Optional[Integer[TT, " *batch seq_len"]] = None,
    ):
        """
        Args:
            pred: predicted tokens
            target: target tokens
            pred_mask: True for positions that predicted.
        """
        if pred_mask is None:
            pred_mask = torch.ones_like(pred, dtype=torch.bool)
        temp = (pred == target) * pred_mask
        correct = temp.sum(dim=-1).to(torch.float32)  # shape (*batch)
        total = pred_mask.sum(dim=-1).to(torch.float32)  # shape (*batch)
        acc = correct / total  # shape (*batch)
        super().update(acc)


# endregion: Specialized metrics for Star Graphs
#######################################################################

# TODO: Can use a common mixin for StarGraphs


class ILMLightningModuleForStarGraphs(ILMLightningModule):
    def _get_generated_length(
        self, pred_ids: Integer[TT, " *batch seq_len"]
    ) -> Integer[TT, " *batch"]:
        pad_token_id = int(cast(int, self.tokenizer.pad_token_id))
        mask_token_id = int(cast(int, self.tokenizer.mask_token_id))
        remove = torch.tensor(
            [mask_token_id, pad_token_id], device=pred_ids.device
        )
        return torch.isin(pred_ids, remove, invert=True).sum(dim=-1)

    def setup_metrics(self):
        super().setup_metrics()
        self.val_generated_length_rmse = MeanSquaredError(squared=False)
        self.test_generated_length_rmse = MeanSquaredError(squared=False)
        self.val_generated_length_mape = MeanAbsolutePercentageError()
        self.test_generated_length_mape = MeanAbsolutePercentageError()
        self.val_exact_match = ExactMatch()
        self.test_exact_match = ExactMatch()
        self.val_hamming_accuracy = HammingAccuracy()
        self.test_hamming_accuracy = HammingAccuracy()

    def update_val_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            clean_pred_ids = self.predictor.clean_up_pred_ids(loss_dict["ids"])
            generated_length = (
                clean_pred_ids != self.tokenizer.pad_token_id
            ).sum(dim=-1)
            actual_length = batch["attention_mask"].sum(dim=-1)
            # Truncate to target length
            clean_pred_ids = clean_pred_ids[
                ..., : batch["input_ids"].shape[-1]
            ]
            self.val_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.val_generated_length_mape.update(
                generated_length, actual_length
            )
            self.val_exact_match.update(
                clean_pred_ids,
                batch["input_ids"],
                pred_length=generated_length,
                target_length=actual_length,
            )
            self.val_hamming_accuracy.update(
                clean_pred_ids, batch["input_ids"], pred_mask=batch["drop"]
            )
        elif dataloader_name == "lm":
            pass  # parent class handles this
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_val_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_val_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "val/generated_length_rmse": self.val_generated_length_rmse,
                    "val/generated_length_mape": self.val_generated_length_mape,
                    "val/exact_match": self.val_exact_match,
                    "val/hamming_accuracy": self.val_hamming_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def update_test_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            clean_pred_ids = self.predictor.clean_up_pred_ids(loss_dict["ids"])
            generated_length = (
                clean_pred_ids != self.tokenizer.pad_token_id
            ).sum(dim=-1)
            actual_length = batch["attention_mask"].sum(dim=-1)
            clean_pred_ids = clean_pred_ids[
                ..., : batch["input_ids"].shape[-1]
            ]
            self.test_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.test_generated_length_mape.update(
                generated_length, actual_length
            )
            self.test_exact_match.update(clean_pred_ids, batch["input_ids"])
            self.test_hamming_accuracy.update(
                clean_pred_ids, batch["input_ids"], pred_mask=batch["drop"]
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_test_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_test_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "test/generated_length_rmse": self.test_generated_length_rmse,
                    "test/generated_length_mape": self.test_generated_length_mape,
                    "test/exact_match": self.test_exact_match,
                    "test/hamming_accuracy": self.test_hamming_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    @rank_zero_only
    def log_predictions(
        self,
        batch: ILMBatch,
        preds: ILMPredictionDict,
        split: Literal["train", "val", "test"],
        dataloader_name: str,
    ):
        step = self.trainer.global_step or 0
        epoch = self.trainer.current_epoch or 0
        file_path = (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger_ = [
            l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")
        ]
        text = []  # list of rows
        ground_truth_text = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )

        with open(file_path, "a") as f:
            for dict_ in self.predictor.to_dict(batch, preds):
                text.append(dict_["text"])
                f.write(json.dumps(dict_) + "\n")
        # only log one set of predictions per eval run.
        if (
            self.trainer.global_step
            > self.last_global_step_logged_at_which_logged_predictions
        ):
            n_rows = 10
            for logger_ in logger_:
                logger_.log_text(
                    f"{split}/{dataloader_name}",
                    ["generated", "truth"],  # column names
                    [
                        [_text, _gt]
                        for _text, _gt in zip(
                            text[:n_rows], ground_truth_text[:n_rows]
                        )
                    ],  # rows
                    step=self.trainer.global_step,
                )
            self.last_global_step_logged_at_which_logged_predictions = (
                self.trainer.global_step
            )


class ILMWithStoppingClassificationLightningModuleForStarGraphs(
    ILMWithStoppingClassificationLightningModule
):
    def _get_generated_length(
        self, pred_ids: Integer[TT, " *batch seq_len"]
    ) -> Integer[TT, " *batch"]:
        pad_token_id = int(cast(int, self.tokenizer.pad_token_id))
        mask_token_id = int(cast(int, self.tokenizer.mask_token_id))
        remove = torch.tensor(
            [mask_token_id, pad_token_id], device=pred_ids.device
        )
        return torch.isin(pred_ids, remove, invert=True).sum(dim=-1)

    def setup_metrics(self):
        super().setup_metrics()
        self.val_generated_length_rmse = MeanSquaredError(squared=False)
        self.test_generated_length_rmse = MeanSquaredError(squared=False)
        self.val_generated_length_mape = MeanAbsolutePercentageError()
        self.test_generated_length_mape = MeanAbsolutePercentageError()
        self.val_exact_match = ExactMatch()
        self.test_exact_match = ExactMatch()
        self.val_hamming_accuracy = HammingAccuracy()
        self.test_hamming_accuracy = HammingAccuracy()

    def update_val_metrics(
        self,
        batch: ILMBatch,
        loss_dict: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            clean_pred_ids = self.predictor.clean_up_pred_ids(loss_dict["ids"])
            generated_length = (
                clean_pred_ids != self.tokenizer.pad_token_id
            ).sum(dim=-1)
            actual_length = batch["attention_mask"].sum(dim=-1)
            # Truncate to target length
            clean_pred_ids = clean_pred_ids[
                ..., : batch["input_ids"].shape[-1]
            ]
            self.val_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.val_generated_length_mape.update(
                generated_length, actual_length
            )
            self.val_exact_match.update(
                clean_pred_ids,
                batch["input_ids"],
                pred_length=generated_length,
                target_length=actual_length,
            )
            self.val_hamming_accuracy.update(
                clean_pred_ids, batch["input_ids"], pred_mask=batch["drop"]
            )
        elif dataloader_name == "lm":
            pass  # parent class handles this
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_val_metrics(
        self,
        batch: ILMBatch,
        metrics: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_val_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "val/generated_length_rmse": self.val_generated_length_rmse,
                    "val/generated_length_mape": self.val_generated_length_mape,
                    "val/exact_match": self.val_exact_match,
                    "val/hamming_accuracy": self.val_hamming_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def update_test_metrics(
        self,
        batch: ILMBatch,
        loss_dict: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            clean_pred_ids = self.predictor.clean_up_pred_ids(loss_dict["ids"])
            generated_length = (
                clean_pred_ids != self.tokenizer.pad_token_id
            ).sum(dim=-1)
            actual_length = batch["attention_mask"].sum(dim=-1)
            clean_pred_ids = clean_pred_ids[
                ..., : batch["input_ids"].shape[-1]
            ]
            self.test_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.test_generated_length_mape.update(
                generated_length, actual_length
            )
            self.test_exact_match.update(clean_pred_ids, batch["input_ids"])
            self.test_hamming_accuracy.update(
                clean_pred_ids, batch["input_ids"], pred_mask=batch["drop"]
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_test_metrics(
        self,
        batch: ILMBatch,
        metrics: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_test_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "test/generated_length_rmse": self.test_generated_length_rmse,
                    "test/generated_length_mape": self.test_generated_length_mape,
                    "test/exact_match": self.test_exact_match,
                    "test/hamming_accuracy": self.test_hamming_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    @rank_zero_only
    def log_predictions(
        self,
        batch: ILMBatch,
        preds: ILMPredictionDict,
        split: Literal["train", "val", "test"],
        dataloader_name: str,
    ):
        step = self.trainer.global_step or 0
        epoch = self.trainer.current_epoch or 0
        file_path = (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger_ = [
            l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")
        ]
        text = []  # list of rows
        ground_truth_text = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )

        with open(file_path, "a") as f:
            for dict_ in self.predictor.to_dict(batch, preds):
                text.append(dict_["text"])
                f.write(json.dumps(dict_) + "\n")
        # only log one set of predictions per eval run.
        if (
            self.trainer.global_step
            > self.last_global_step_logged_at_which_logged_predictions
        ):
            n_rows = 10
            for logger_ in logger_:
                logger_.log_text(
                    f"{split}/{dataloader_name}",
                    ["generated", "truth"],  # column names
                    [
                        [_text, _gt]
                        for _text, _gt in zip(
                            text[:n_rows], ground_truth_text[:n_rows]
                        )
                    ],  # rows
                    step=self.trainer.global_step,
                )
            self.last_global_step_logged_at_which_logged_predictions = (
                self.trainer.global_step
            )


# endregion: IDLM for Star Graphs
################################################################################

################################################################################
# region: ILM for Countdown


class CountdownAccuracy(MeanMetric):
    def __init__(self, *args, num_expressions: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_expressions = num_expressions

    def update(
        self,
        input_numbers: List[List[int]],
        predicted_output_expressions: List[List[Tuple[str, str]]],
    ):
        """
        Args:
            input_numbers: List[List[int]], Numbers in the input
            predicted_output_numbers: List[List[int]], Numbers in the output
            predicted_output_expressions: List[List[Tuple[str, str]]], Expressions in the output
        """
        acc = []
        for _input_numbers, _predicted_expressions in zip(
            input_numbers, predicted_output_expressions
        ):
            _final_target_number = _input_numbers[-1]
            try:
                _final_predicted_number = int(_predicted_expressions[-1][-1])
            except ValueError:
                acc.append(0)
                continue
            if len(_predicted_expressions) != self.num_expressions:
                acc.append(0)
                continue
            correct_expressions = []
            for _pred_lhs, _pred_rhs in _predicted_expressions:
                try:
                    _correct = eval(_pred_lhs) == eval(_pred_rhs)
                except (ZeroDivisionError, SyntaxError):
                    _correct = False
                correct_expressions.append(_correct)
            if (
                all(correct_expressions)
                and _final_target_number == _final_predicted_number
            ):
                acc.append(1)
            else:
                acc.append(0)
        _acc = torch.tensor(acc, device=self.device)
        super().update(_acc)


class ILMWithStoppingClassificationLightningModuleForCountdown(
    ILMWithStoppingClassificationLightningModule
):
    def __init__(self, *args, num_expressions: Optional[int] = None, **kwargs):
        if num_expressions is None:
            if hasattr(self, "datamodule") and self.datamodule is not None:
                num_expressions = self.datamodule.num_expressions
            elif kwargs.get("datamodule", None) is not None:
                num_expressions = kwargs["datamodule"].num_expressions
            else:
                raise ValueError("num_expressions must be provided")

        self.num_expressions = num_expressions
        super().__init__(*args, **kwargs)

    def _get_generated_length(
        self, pred_ids: Integer[TT, " *batch seq_len"]
    ) -> Integer[TT, " *batch"]:
        pad_token_id = int(cast(int, self.tokenizer.pad_token_id))
        mask_token_id = int(cast(int, self.tokenizer.mask_token_id))
        remove = torch.tensor(
            [mask_token_id, pad_token_id], device=pred_ids.device
        )
        return torch.isin(pred_ids, remove, invert=True).sum(dim=-1)

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if flags.DEBUG_CD_NAN:
            for name, param in self.model.named_parameters():
                if param.isnan().any():
                    message = f"NaN parameter found at validation epoch end at epoch={self.trainer.current_epoch} at step={self.trainer.global_step}"
                    raise ValueError(message)

    def on_training_batch_start(self, batch: ILMBatch, batch_idx: int) -> None:
        if flags.DEBUG_CD_NAN:
            for name, param in self.model.named_parameters():
                if param.isnan().any():
                    message = f"NaN parameter found at training batch start at epoch={self.trainer.current_epoch} at step={self.trainer.global_step}"
                    raise ValueError(message)

    def setup_metrics(self):
        super().setup_metrics()
        self.val_generated_length_rmse = MeanSquaredError(squared=False)
        self.test_generated_length_rmse = MeanSquaredError(squared=False)
        self.val_generated_length_mape = MeanAbsolutePercentageError()
        self.test_generated_length_mape = MeanAbsolutePercentageError()
        self.val_exact_match = ExactMatch()
        self.test_exact_match = ExactMatch()
        self.val_hamming_accuracy = HammingAccuracy()
        self.test_hamming_accuracy = HammingAccuracy()
        self.val_countdown_accuracy = CountdownAccuracy(
            num_expressions=self.num_expressions
        )
        self.test_countdown_accuracy = CountdownAccuracy(
            num_expressions=self.num_expressions
        )

    def update_val_metrics(
        self,
        batch: ILMBatch,
        loss_dict: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            clean_pred_ids = self.predictor.clean_up_pred_ids(loss_dict["ids"])
            generated_length = (
                clean_pred_ids != self.tokenizer.pad_token_id
            ).sum(dim=-1)
            actual_length = batch["attention_mask"].sum(dim=-1)
            # Truncate to target length
            clean_pred_ids = clean_pred_ids[
                ..., : batch["input_ids"].shape[-1]
            ]
            self.val_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.val_generated_length_mape.update(
                generated_length, actual_length
            )
            self.val_exact_match.update(
                clean_pred_ids,
                batch["input_ids"],
                pred_length=generated_length,
                target_length=actual_length,
            )
            self.val_hamming_accuracy.update(
                clean_pred_ids, batch["input_ids"], pred_mask=batch["drop"]
            )
            coverted = self.tokenizer.batch_convert_ids_to_expressions(clean_pred_ids)  # type: ignore
            self.val_countdown_accuracy.update(
                coverted["input_numbers"],
                coverted["output_expressions"],
            )
        elif dataloader_name == "lm":
            pass  # parent class handles this
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_val_metrics(
        self,
        batch: ILMBatch,
        metrics: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_val_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "val/generated_length_rmse": self.val_generated_length_rmse,
                    "val/generated_length_mape": self.val_generated_length_mape,
                    "val/exact_match": self.val_exact_match,
                    "val/hamming_accuracy": self.val_hamming_accuracy,
                    "val/countdown_accuracy": self.val_countdown_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def update_test_metrics(
        self,
        batch: ILMBatch,
        loss_dict: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            clean_pred_ids = self.predictor.clean_up_pred_ids(loss_dict["ids"])
            generated_length = (
                clean_pred_ids != self.tokenizer.pad_token_id
            ).sum(dim=-1)
            actual_length = batch["attention_mask"].sum(dim=-1)
            clean_pred_ids = clean_pred_ids[
                ..., : batch["input_ids"].shape[-1]
            ]
            self.test_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.test_generated_length_mape.update(
                generated_length, actual_length
            )
            self.test_exact_match.update(clean_pred_ids, batch["input_ids"])
            self.test_hamming_accuracy.update(
                clean_pred_ids, batch["input_ids"], pred_mask=batch["drop"]
            )
            coverted = self.tokenizer.batch_convert_ids_to_expressions(clean_pred_ids)  # type: ignore
            self.test_countdown_accuracy.update(
                coverted["input_numbers"],
                coverted["output_expressions"],
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_test_metrics(
        self,
        batch: ILMBatch,
        metrics: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_test_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "test/generated_length_rmse": self.test_generated_length_rmse,
                    "test/generated_length_mape": self.test_generated_length_mape,
                    "test/exact_match": self.test_exact_match,
                    "test/hamming_accuracy": self.test_hamming_accuracy,
                    "test/countdown_accuracy": self.test_countdown_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    @rank_zero_only
    def log_predictions(
        self,
        batch: ILMBatch,
        preds: ILMPredictionDict,
        split: Literal["train", "val", "test"],
        dataloader_name: str,
    ):
        step = self.trainer.global_step or 0
        epoch = self.trainer.current_epoch or 0
        file_path = (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger_ = [
            l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")
        ]
        text = []  # list of rows
        ground_truth_text = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )

        with open(file_path, "a") as f:
            for dict_ in self.predictor.to_dict(batch, preds):
                text.append(dict_["text"])
                f.write(json.dumps(dict_) + "\n")
        # only log one set of predictions per eval run.
        if (
            self.trainer.global_step
            > self.last_global_step_logged_at_which_logged_predictions
        ):
            n_rows = 10
            for logger_ in logger_:
                logger_.log_text(
                    f"{split}/{dataloader_name}",
                    ["generated", "truth"],  # column names
                    [
                        [_text, _gt]
                        for _text, _gt in zip(
                            text[:n_rows], ground_truth_text[:n_rows]
                        )
                    ],  # rows
                    step=self.trainer.global_step,
                )
            self.last_global_step_logged_at_which_logged_predictions = (
                self.trainer.global_step
            )


# endregion: ILM for Countdown
################################################################################

################################################################################
# region: ILM for Zebra


class ZebraIoU(MeanMetric):
    def update(
        self,
        pred_solution_sets: List[Set[Tuple[int, int, int]]],
        target_solution_sets: List[Set[Tuple[int, int, int]]],
    ):
        ious = []
        for _pred_solution_set, _target_solution_set in zip(
            pred_solution_sets, target_solution_sets
        ):
            # Compute IOU
            intersection = _pred_solution_set & _target_solution_set
            union = _pred_solution_set | _target_solution_set
            iou = len(intersection) / len(union)
            ious.append(iou)
        _ious = torch.tensor(ious, device=self.device)
        super().update(_ious)


class ZebraAccuracy(MeanMetric):
    def update(
        self,
        pred_solution_sets: List[Set[Tuple[int, int, int]]],
        target_solution_sets: List[Set[Tuple[int, int, int]]],
    ):
        acc = []
        for _pred_solution_set, _target_solution_set in zip(
            pred_solution_sets, target_solution_sets
        ):
            acc.append(float(_pred_solution_set == _target_solution_set))

        _acc = torch.tensor(acc, device=self.device)
        super().update(_acc)


# ILMWithStoppingClassificationLightningModuleForZebra = (
#     ILMWithStoppingClassificationLightningModuleForStarGraphs
# )


class ILMWithStoppingClassificationLightningModuleForZebra(
    ILMWithStoppingClassificationLightningModule
):

    def _get_generated_length(
        self, pred_ids: Integer[TT, " *batch seq_len"]
    ) -> Integer[TT, " *batch"]:
        """Same as StarGraphs."""
        pad_token_id = int(cast(int, self.tokenizer.pad_token_id))
        mask_token_id = int(cast(int, self.tokenizer.mask_token_id))
        remove = torch.tensor(
            [mask_token_id, pad_token_id], device=pred_ids.device
        )
        return torch.isin(pred_ids, remove, invert=True).sum(dim=-1)

    def setup_metrics(self):
        super().setup_metrics()
        self.val_exact_match = ZebraAccuracy()
        self.test_exact_match = ZebraAccuracy()
        self.val_iou = ZebraIoU()
        self.test_iou = ZebraIoU()
        # Let the length metrics run on raw length
        self.val_generated_length_rmse = MeanSquaredError(squared=False)
        self.test_generated_length_rmse = MeanSquaredError(squared=False)
        self.val_generated_length_mape = MeanAbsolutePercentageError()
        self.test_generated_length_mape = MeanAbsolutePercentageError()

    def update_val_metrics(
        self,
        batch: ILMBatch,
        loss_dict: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            clean_pred_ids = self.predictor.clean_up_pred_ids(loss_dict["ids"])
            generated_length = (
                clean_pred_ids != self.tokenizer.pad_token_id
            ).sum(dim=-1)
            actual_length = batch["attention_mask"].sum(dim=-1)
            # Truncate to target length
            clean_pred_ids = clean_pred_ids[
                ..., : batch["input_ids"].shape[-1]
            ]
            self.val_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.val_generated_length_mape.update(
                generated_length, actual_length
            )
            pred_solution_sets = self.tokenizer.get_solution_sets_from_batch(clean_pred_ids)  # type: ignore
            target_solution_sets = self.tokenizer.get_solution_sets_from_batch(batch["input_ids"])  # type: ignore
            self.val_exact_match.update(
                pred_solution_sets,
                target_solution_sets,
            )
            self.val_iou.update(
                pred_solution_sets,
                target_solution_sets,
            )
        elif dataloader_name == "lm":
            pass  # parent class handles this
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_val_metrics(
        self,
        batch: ILMBatch,
        metrics: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_val_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "val/generated_length_rmse": self.val_generated_length_rmse,
                    "val/generated_length_mape": self.val_generated_length_mape,
                    "val/exact_match": self.val_exact_match,
                    "val/iou": self.val_iou,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def update_test_metrics(
        self,
        batch: ILMBatch,
        loss_dict: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            clean_pred_ids = self.predictor.clean_up_pred_ids(loss_dict["ids"])
            generated_length = (
                clean_pred_ids != self.tokenizer.pad_token_id
            ).sum(dim=-1)
            actual_length = batch["attention_mask"].sum(dim=-1)
            clean_pred_ids = clean_pred_ids[
                ..., : batch["input_ids"].shape[-1]
            ]
            self.test_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.test_generated_length_mape.update(
                generated_length, actual_length
            )
            pred_solution_sets = self.tokenizer.get_solution_sets_from_batch(clean_pred_ids)  # type: ignore
            target_solution_sets = self.tokenizer.get_solution_sets_from_batch(batch["input_ids"])  # type: ignore
            self.test_exact_match.update(
                pred_solution_sets,
                target_solution_sets,
            )
            self.test_iou.update(
                pred_solution_sets,
                target_solution_sets,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_test_metrics(
        self,
        batch: ILMBatch,
        metrics: ILMWithStoppingClassificationLossDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_test_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "test/generated_length_rmse": self.test_generated_length_rmse,
                    "test/generated_length_mape": self.test_generated_length_mape,
                    "test/exact_match": self.test_exact_match,
                    "test/iou": self.test_iou,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    @rank_zero_only
    def log_predictions(
        self,
        batch: ILMBatch,
        preds: ILMPredictionDict,
        split: Literal["train", "val", "test"],
        dataloader_name: str,
    ):
        step = self.trainer.global_step or 0
        epoch = self.trainer.current_epoch or 0
        file_path = (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger_ = [
            l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")
        ]
        text = []  # list of rows
        ground_truth_text = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        self.fields_to_keep_in_output = ["text", "truth"]

        def filter_fields(out_dict):
            if self.fields_to_keep_in_output is None:
                return out_dict
            return {
                k: v
                for k, v in out_dict.items()
                if k in self.fields_to_keep_in_output
            }

        def _to_dict(batch, preds):
            _out_dict = self.predictor.to_dict(batch, preds)
            # add the ground truth text
            for i, _one_out_dict in enumerate(_out_dict):
                _one_out_dict["truth"] = ground_truth_text[i]
            return _out_dict

        with open(file_path, "a") as f:
            for dict_ in _to_dict(batch, preds):
                text.append(dict_["text"])
                f.write(json.dumps(filter_fields(dict_)) + "\n")
        # only log one set of predictions per eval run.
        if (
            self.trainer.global_step
            > self.last_global_step_logged_at_which_logged_predictions
        ):
            n_rows = 10
            for logger_ in logger_:
                logger_.log_text(
                    f"{split}/{dataloader_name}",
                    ["generated", "truth"],  # column names
                    [
                        [_text, _gt]
                        for _text, _gt in zip(
                            text[:n_rows], ground_truth_text[:n_rows]
                        )
                    ],  # rows
                    step=self.trainer.global_step,
                )
            self.last_global_step_logged_at_which_logged_predictions = (
                self.trainer.global_step
            )


# endregion: ILM for Zebra
################################################################################

################################################################################
# region: ILM for CFG

from .idlm_v2 import Parsable


class ILMLightningModuleForCFG(ILMWithStoppingClassificationLightningModule):
    def setup_metrics(self):
        super().setup_metrics()
        self.val_parsable = Parsable(self.datamodule.parser, self.tokenizer)  # type: ignore
        self.test_parsable = Parsable(self.datamodule.parser, self.tokenizer)  # type: ignore

    def update_val_metrics(
        self, batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
    ):
        super().update_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            computed = self.val_parsable._just_compute(loss_dict["text"])
            loss_dict["per_sample_parsable"] = computed.bool().tolist()
            self.val_parsable.update(loss_dict["text"], computed)

    def log_val_metrics(
        self, batch, metrics, batch_idx, dataloader_idx, dataloader_name
    ):
        super().log_val_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {"val/parsable": self.val_parsable},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )

    def update_test_metrics(
        self, batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
    ):
        super().update_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            computed = self.test_parsable._just_compute(batch["text"])
            loss_dict["per_sample_parsable"] = computed.bool().tolist()
            self.test_parsable.update(batch["text"], computed)

    def log_test_metrics(
        self, batch, metrics, batch_idx, dataloader_idx, dataloader_name
    ):
        super().log_test_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {"test/parsable": self.test_parsable},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )

    @rank_zero_only
    def log_predictions(
        self,
        batch: ILMBatch,
        preds: ILMPredictionDict,
        split: Literal["train", "val", "test"],
        dataloader_name: str,
    ):
        step = self.trainer.global_step or 0
        epoch = self.trainer.current_epoch or 0
        file_path = (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger_ = [
            l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")
        ]

        per_sample_parsable = preds.get("per_sample_parsable", None)
        fields_to_keep_in_output = list(
            set(
                ["text", "parsable"]
                + (
                    self.fields_to_keep_in_output
                    if self.fields_to_keep_in_output is not None
                    else []
                )
            )
        )

        def filter_fields(out_dict):
            if fields_to_keep_in_output is None:
                return out_dict
            return {
                k: v
                for k, v in out_dict.items()
                if k in fields_to_keep_in_output
            }

        def _to_dict(batch, preds):
            _out_dict = self.predictor.to_dict(batch, preds)
            for i, _one_out_dict in enumerate(_out_dict):
                if per_sample_parsable is not None:
                    _one_out_dict["parsable"] = bool(per_sample_parsable[i])
            return _out_dict

        text = []  # list of rows
        parsable = []  # list of rows
        with open(file_path, "a") as f:
            for dict_ in _to_dict(batch, preds):
                text.append(dict_["text"])
                parsable.append(dict_["parsable"])
                f.write(json.dumps(filter_fields(dict_)) + "\n")

        # only log one set of predictions per eval run.
        if (
            self.trainer.global_step
            > self.last_global_step_logged_at_which_logged_predictions
        ):
            n_rows = 10
            for logger_ in logger_:
                logger_.log_text(
                    f"{split}/{dataloader_name}",
                    ["generated", "parsable"],  # column names
                    [
                        [_text, _parsable]
                        for _text, _parsable in zip(
                            text[:n_rows], parsable[:n_rows]
                        )
                    ],  # rows
                    step=self.trainer.global_step,
                )
            self.last_global_step_logged_at_which_logged_predictions = (
                self.trainer.global_step
            )


# endregion: ILM for CFG

################################################################################
