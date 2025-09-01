# v2
import collections
from copy import deepcopy
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Protocol,
    Tuple,
    Union,
    TypedDict,
    Mapping,
    Any,
    Optional,
    cast,
)
from huggingface_hub import get_token
from jaxtyping import Integer, Float, Bool
from torch import Tensor as TT
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import collate_tensor_fn
from numpy import ndarray as NA
import numpy as np
import random
import torch
from transformers import (
    BatchEncoding,
    PreTrainedTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
    PreTrainedTokenizerBase,
)
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
import lightning as L
from pcdd.utils.rank_zero import rank_zero_only
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.rank_zero import RankedLogger
from tokenizers import processors

logger = RankedLogger(__name__, rank_zero_only=True)

################################################################################
# region: Types


class Tokenizer(Protocol):
    mask_token_id: int
    pad_token_id: int
    cls_token_id: int
    eos_token_id: int
    bos_token_id: int
    mask_token: str
    pad_token: str
    cls_token: str
    eos_token: str
    bos_token: str

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding: ...

    @property
    def vocab_size(self) -> int: ...

    def __len__(self) -> int: ...

    def decode(
        self,
        token_ids: Union[List[int], Integer[TT, " seq_len"]],
        skip_special_tokens: bool = True,
    ) -> str: ...

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], Integer[TT, " batch seq_len"]],
        skip_special_tokens: bool = True,
    ) -> List[str]: ...

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]: ...

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]: ...


class BaseBatch(TypedDict):
    """Dict with the keys that are present in input batches for all models.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): Can depend on the model type.
            For ILM and IDLM: 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[
        TT, " batch seq_len"
    ]  # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
    # CLEANUP: There is no use for token_type_ids for ARLM and MLM. Instead we should only have constraint tensor.


class MLMBatch(BaseBatch):
    """Input to the MLM.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    drop: Integer[TT, " batch seq_len"]


class MDLMBatch(BaseBatch):
    """MDLMBatch

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        drop (Integer[TT, " batch seq_len"]): 1 for tokens that are dropped.
        t (Integer[TT, " batch"]): The time step.
        noise_rate (Float[TT, " batch"]): The noise rate.
        total_noise (Integer[TT, " batch"]): The total number of noise tokens.
        constraint (Optional[Bool[TT, " batch seq_len"]]): 1 for tokens that should not be predicted. Mostly used during prediction only.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Optional[Integer[TT, " batch seq_len"]]
    drop: Integer[TT, " batch seq_len"]
    t: Integer[TT, " batch"]
    noise_rate: Float[TT, " batch"]
    total_noise: Integer[TT, " batch"]
    constraint: Optional[Bool[TT, " batch seq_len"]]


class IDLMBatch(BaseBatch):
    """Input to the LossFunction Callable.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        drop (Integer[TT, " batch seq_len"]): 1 for tokens that are dropped.
        target_ids (Integer[TT, " batch seq_len vocab_size"]): The target ids to the model.
        t (Integer[TT, " batch"]): The time step.
        noise_rate (Float[TT, " batch"]): The noise rate.
        total_noise (Integer[TT, " batch"]): The total number of noise tokens.
        constraint (Optional[Bool[TT, " batch seq_len"]]): 1 for positions out of which there should be no prediction.
            Mostly used during prediction only.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    drop: Integer[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len vocab_size"]
    t: Integer[TT, " batch"]
    noise_rate: Float[TT, " batch"]
    total_noise: Integer[TT, " batch"]
    constraint: Optional[
        Bool[TT, " batch seq_len"]
    ]  # to suppress prediction from specific token positions. 1 means suppress.
    # Mostly used during prediction only.


class ILMBatch(BaseBatch):
    """Input to the ILM.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        drop (Bool[TT, " batch seq_len"]): 1 for tokens that are dropped.
        target_ids (Integer[TT, " batch seq_len vocab_size"]): The target ids to the model.
        constraint (Optional[Bool[TT, " batch seq_len"]]): 1 for tokens that should not be predicted. Mostly used during prediction only.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[TT, " batch seq_len"]
    drop: Bool[TT, " batch seq_len"]
    target_ids: Integer[TT, " batch seq_len vocab_size"]
    constraint: Optional[Bool[TT, " batch seq_len"]]


class ITBatch(ILMBatch):
    counts: Integer[TT, " batch seq_len"]


class ARLMBatch(BaseBatch):
    """ARLMBatch

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
        drop (Integer[TT, " batch seq_len"]): 1 for tokens that are dropped.
        constraint (Optional[Bool[TT, " batch seq_len"]]): 1 for tokens that should not be predicted. Mostly used during prediction only.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Optional[Integer[TT, " batch seq_len"]]
    drop: Integer[TT, " batch seq_len"]
    constraint: Optional[Bool[TT, " batch seq_len"]]


class DataLoaderKwargs(TypedDict):
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool


class Processor(Protocol):
    def __call__(
        self,
        examples: Mapping[str, List[str]],
        tokenizer: Tokenizer,
        **kwargs,
    ) -> Mapping[str, List[str]]: ...

    """Applied on a dataset using `dataset.map`.
    Args:
        examples: A dictionary containing a batch of data.
        tokenizer: The tokenizer to use for tokenization.
        **kwargs: Additional keyword arguments.
    Returns:
        A dictionary containing the processed data, typically tokenized and grouped/padded.
    """


class Collator(Protocol):
    tokenizer: Tokenizer
    block_size: int
    noise_schedule: NoiseSchedule

    def __call__(
        self,
        batch: List[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


# endregion: Types
################################################################################


################################################################################
# region: Base DataModule


class BaseDataModule(L.LightningDataModule):
    """
    Base class for all datamodules.
    """

    train_dataloader_names: Dict[int, str]
    """A mapping the dataloader_id to the name.[Required]"""
    val_dataloader_names: Dict[int, str]
    """A mapping the dataloader_id to the name.[Required]"""
    test_dataloader_names: Dict[int, str]
    """A mapping the dataloader_id to the name.[Required]"""
    predict_dataloader_names: Dict[int, str]
    """A mapping the dataloader_id to the name.[Required]"""
    train_dataloader_kwargs: DataLoaderKwargs
    """DataLoaderKwargs for the train dataloader.[Required]"""
    val_dataloader_kwargs: DataLoaderKwargs
    """DataLoaderKwargs for the val dataloader.[Required]"""
    test_dataloader_kwargs: DataLoaderKwargs
    """DataLoaderKwargs for the test dataloader.[Required]"""
    predict_dataloader_kwargs: DataLoaderKwargs
    """DataLoaderKwargs for the predict dataloader.[Required]"""
    tokenizer: Tokenizer
    """The tokenizer.[Required]"""
    global_batch_size: int
    """The global batch size.[Required]"""
    train_dataset_lm: Optional[IterableDataset] = None

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__()  # LightningDataModule.__init__() does not take any arguments
        self.prepare_data_per_node = False
        self.train_dataloader_names = {}
        self.val_dataloader_names = {}
        self.test_dataloader_names = {}
        self.predict_dataloader_names = {}

    def print_batch(
        self,
        batch: BaseBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        """Required to print train and validation batches at the beginning of the epoch."""
        raise NotImplementedError("Not implemented")

    def _check_grad_accum(self):
        if self.trainer is None:
            logger.warning("Trainer is not setup. Cannot check grad accum.")
            return
        if self._is_ddp():
            num_nodes = self.trainer.num_nodes
            num_gpus_per_node = self.trainer.num_devices
            accum_steps = self.trainer.accumulate_grad_batches
            logger.info(
                f"global_batch_size: {self.global_batch_size} | num_nodes: {num_nodes} | num_gpus_per_node: {num_gpus_per_node} | accum_steps: {accum_steps}"
            )
            if (
                self.global_batch_size
                % (
                    self.train_dataloader_kwargs["batch_size"]
                    * num_nodes
                    * num_gpus_per_node
                    * accum_steps
                )
                != 0
            ):
                raise ValueError(
                    f"Global batch size ({self.global_batch_size}) is not equal to "
                    f"per_device_batch_size ({self.train_dataloader_kwargs['batch_size']}) * num_nodes ({num_nodes}) * num_gpus_per_node ({num_gpus_per_node}) * accum_steps ({accum_steps})."
                )

    def _is_ddp(self) -> bool:
        if self.trainer is not None:
            strategy = self.trainer.strategy
            if isinstance(strategy, DDPStrategy):
                return True
            elif isinstance(strategy, SingleDeviceStrategy):
                return False
            else:
                raise ValueError(
                    f"Dataloader does not support {type(strategy)} strategy"
                )
        else:
            logger.warning(
                "Tried to detect DDP strategy before trainer was set."
                " Are you calling `LightningDataModule.*_dataloader()` methods manually?"
                " Make sure you know what you are doing!"
            )
            return False

    def set_epoch(self, epoch: int) -> None:
        if self.train_dataset_lm is not None:
            self.train_dataset_lm.set_epoch(epoch)


# endregion: Base DataModule
################################################################################


################################################################################
# region: Tokenizers


class IDLMTokenizerMixin:
    """Overrides the two key methods.
    Should be used as a mixin with mro order:
    class IDLMTokenizer(IDLMTokenizerMixin, PreTrainedTokenizerFast):
        pass
    Note:
      Make sure to call `post_creation` after initializing the tokenizer.
    """

    @property
    def full_vocab_size(self) -> int:
        return self.__len__()

    def post_creation(self):
        """Check the presence of the special tokens and update the post processor."""
        for special_token in [
            "eos_token",
            "bos_token",
            "cls_token",
            "pad_token",
            "mask_token",
            "sep_token",
            "unk_token",
        ]:
            if (token := getattr(self, special_token)) is None:
                raise ValueError(f"{special_token} is not set")
        if isinstance(self, PreTrainedTokenizerFast):
            self.update_post_processor()

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # token_ids_1 is assumed to be a prefix
        if token_ids_1 is not None:
            return (
                [self.cls_token_id]  # type: ignore
                + [self.bos_token_id]  # type: ignore
                + token_ids_1
                + token_ids_0
            )  # type: ignore
        else:
            return [self.cls_token_id] + [self.bos_token_id] + token_ids_0  # type: ignore

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # token_ids_1 is assumed to be a prefix
        if token_ids_1 is None:
            return [0, 1] + [2] * len(token_ids_0)  # type: ignore
        else:
            return [0, 1] + [1] * len(token_ids_1) + [2] * len(token_ids_0)  # type: ignore

    def create_post_processor(self) -> processors.TemplateProcessing:
        if (self.cls_token is None) or (self.cls_token_id is None):
            raise ValueError("cls_token is required.")
        if (self.bos_token is None) or (self.bos_token_id is None):
            raise ValueError("bos_token is required.")
        post_processor = processors.TemplateProcessing(
            single=f"{self.cls_token}:0 {self.bos_token}:1 $A:2",
            pair=f"{self.cls_token}:0 {self.bos_token}:1 $B:1 $A:2",
            special_tokens=[
                (self.cls_token, self.cls_token_id),
                (self.bos_token, self.bos_token_id),
            ],
        )
        return post_processor

    def update_post_processor(self) -> None:
        self.post_processor = self.create_post_processor()
        self._tokenizer.post_processor = self.post_processor


class MDLMTokenizerMixin:  # type: ignore

    @property
    def full_vocab_size(self) -> int:
        return self.__len__()

    def post_creation(self):
        """Check the presence of the special tokens and update the post processor."""
        for special_token in [
            "bos_token",
            "cls_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "mask_token",
        ]:
            if (token := getattr(self, special_token)) is None:
                raise ValueError(f"{special_token} is not set")
        if isinstance(self, PreTrainedTokenizerFast):
            self.update_post_processor()

    def create_post_processor(self):
        if (self.eos_token is None) or (self.eos_token_id is None):
            raise ValueError("eos_token is required.")
        if (self.bos_token is None) or (self.bos_token_id is None):
            raise ValueError("bos_token is required.")
        return processors.BertProcessing(
            (self.bos_token, self.bos_token_id),
            (self.eos_token, self.eos_token_id),
        )

    def update_post_processor(self) -> None:
        self.post_processor = self.create_post_processor()
        self._tokenizer.post_processor = self.post_processor

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # token_ids_1 is assumed to be a prefix
        if token_ids_1 is not None:
            return [self.bos_token_id] + token_ids_1 + token_ids_0 + [self.eos_token_id]  # type: ignore
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]  # type: ignore

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is not None:
            return [1] * (len(token_ids_1) + 1) + [2] * len(token_ids_0)  # type: ignore
        # bos + token_ids_0 + eos
        return [2] * (len(token_ids_0) + 2)


class ARLMTokenizerMixin:
    @property
    def full_vocab_size(self) -> int:
        return self.__len__()

    def post_creation(self):
        """Check the presence of the special tokens and update the post processor."""
        for special_token in [
            "eos_token",
            "bos_token",
            "pad_token",
            "mask_token",
        ]:
            if (token := getattr(self, special_token)) is None:
                raise ValueError(f"{special_token} is not set")
        if isinstance(self, PreTrainedTokenizerFast):
            self.update_post_processor()

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # token_ids_1 is assumed to be a prefix
        if token_ids_1 is not None:
            return (
                token_ids_1
                + [self.bos_token_id]
                + token_ids_0
                + [self.eos_token_id]
            )  # type: ignore
        else:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # BOS+EOS=2, prefix=1, non_prefix=2
        # token_ids_1 is assumed to be a prefix
        if token_ids_1 is None:
            return [1] + [2] * len(token_ids_0) + [2]  # type: ignore
        else:
            return [1] * len(token_ids_1) + [1] + [2] * len(token_ids_0) + [2]  # type: ignore

    def create_post_processor(self) -> processors.TemplateProcessing:
        if (self.bos_token is None) or (self.bos_token_id is None):
            raise ValueError("bos_token is required.")
        if (self.eos_token is None) or (self.bos_token_id is None):
            raise ValueError("bos_token is required.")
        post_processor = processors.TemplateProcessing(
            single=f"{self.bos_token}:2 $A:2 {self.eos_token}:2",
            pair=f"$B:1 {self.bos_token}:2 $A:2 {self.eos_token}:2",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )
        return post_processor

    def update_post_processor(self) -> None:
        self.post_processor = self.create_post_processor()
        self._tokenizer.post_processor = self.post_processor


class BertTokenizerForMDLM(MDLMTokenizerMixin, BertTokenizer):  # type: ignore
    def __init__(self, *args, bos_token: str = "[BOS]", eos_token: str = "[EOS]", **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {"bos_token": bos_token, "eos_token": eos_token}
        )


class BertTokenizerForMDLMFast(MDLMTokenizerMixin, BertTokenizerFast):  # type: ignore
    def __init__(self, *args, bos_token: str = "[BOS]", eos_token: str = "[EOS]", **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {"bos_token": bos_token, "eos_token": eos_token}
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class LegacyBertTokenizerForMDLMFast(MDLMTokenizerMixin, BertTokenizerFast):  # type: ignore
    """Uses CLS as BOS and SEP as EOS."""

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        cls_token = self.cls_token
        sep_token = self.sep_token
        assert cls_token is not None
        assert sep_token is not None
        self.add_special_tokens(
            {"bos_token": cls_token, "eos_token": sep_token}
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class BertTokenizerForIDLM(IDLMTokenizerMixin, BertTokenizer):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens({"cls_token": "[CLS]", "bos_token": "[BOS]"})

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class BertTokenizerForIDLMFast(IDLMTokenizerMixin, BertTokenizerFast):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {"cls_token": "[CLS]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


BertTokenizerForILMFast = BertTokenizerForIDLMFast
BertTokenizerForILM = BertTokenizerForIDLM
BertTokenizerForITFast = BertTokenizerForIDLMFast
BertTokenizerForIT = BertTokenizerForIDLM


class GPT2TokenizerForILMFast(IDLMTokenizerMixin, GPT2TokenizerFast):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {
                "cls_token": "<|cls|>",
                "bos_token": "<|bos|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "mask_token": "<|mask|>",
                "sep_token": "<|sep|>",
                "eos_token": "<|endoftext|>",  # original
            }
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class BertTokenizerForARLM(ARLMTokenizerMixin, BertTokenizer):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class BertTokenizerForARLMFast(ARLMTokenizerMixin, BertTokenizerFast):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


# endregion: Tokenizers
################################################################################


################################################################################
# region: Processors


def ids_to_example_fn(
    example: Dict[Literal["token_ids"], List[int]],
    tokenizer: PreTrainedTokenizerBase,
    block_size: Optional[int] = None,
) -> Dict[Literal["input_ids", "attention_mask", "token_type_ids"], List[int]]:
    """Convert raw token_ids to input_ids, attention_mask, and token_type_ids.

    Does:
        1. Calls `tokenizer.build_inputs_with_special_tokens` and `tokenizer.create_token_type_ids_from_sequences`
            to produce `input_ids` and `token_type_ids`.
        2. Creates an `attention_mask` of all ones.

    Does not do:
        1. Padding/truncation.

    Args:
        example: A dictionary with a "token_ids" key, and value which is a list of token ids.
        tokenizer: A tokenizer that implements `PretrainedTokenizerBase` interface.
            Specifically, it should have `build_inputs_with_special_tokens` and
            `create_token_type_ids_from_sequences` methods overridden if the default
            implementations are not correct.
        block_size: The block size to pad/truncate the input_ids to.
    Returns:
        A dictionary with "input_ids", "attention_mask", and "token_type_ids" keys.
    """
    input_ids = tokenizer.build_inputs_with_special_tokens(
        example["token_ids"]
    )
    attention_mask = [1] * len(input_ids)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(
        example["token_ids"]
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


# CLEANUP: Don't need PadTruncateProcessor anymore
class PadTruncateProcessor(Processor):
    def __init__(
        self,
        block_size: int = 128,
        text_key: str = "text",
        return_token_type_ids: bool = True,
    ):
        self.block_size = block_size
        self.text_key = text_key
        self.return_token_type_ids = return_token_type_ids

    def __call__(
        self,
        examples: Mapping[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ) -> Mapping[str, Any]:
        """
        Tokenize and pad/truncate the text to a fixed block size.
        """
        text = examples[self.text_key]
        tokenized = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=self.return_token_type_ids,
        )
        return tokenized


class DefaultEmptyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        empty_text: str = "",
    ):
        """
        Args:
            tokenizer_kwargs: Keyword arguments for the tokenizer.

            empty_text: For MLM, you will want to set the `empty_text` to a sequence of all mask tokens.
        """
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.empty_text = empty_text

    def __iter__(self):
        for _ in range(self.num_examples):
            ex = self.tokenizer(
                self.empty_text,
                add_special_tokens=True,
                **self.tokenizer_kwargs,
            )
            yield ex


class MDLMEmptyDataset(IterableDataset):
    """Generates a dataset where each example is a sequence of mask tokens."""

    def __init__(
        self,
        num_examples: int,
        max_length: int,
        tokenizer: Tokenizer,
    ):
        """
        Args:
            num_examples: The number of examples to generate.
            max_length: The max_length of the dataset.
        """
        self.num_examples = num_examples
        self.max_length = max_length
        ids = [tokenizer.mask_token_id] * max_length
        ids_ = tokenizer.build_inputs_with_special_tokens(ids)
        diff = len(ids_) - max_length
        self.ids = [tokenizer.mask_token_id] * (max_length - diff)
        self.tokenizer = tokenizer

    def __iter__(self):
        for _ in range(self.num_examples):
            ex = deepcopy(self.ids)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(ex)
            attention_mask = [1] * len(input_ids)
            token_type_ids = (
                self.tokenizer.create_token_type_ids_from_sequences(ex)
            )
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }


class ARLMEmptyDataset(IterableDataset):
    """Generates a dataset where each example is a sequence of mask tokens."""

    def __init__(
        self,
        num_examples: int,
        tokenizer: Tokenizer,
    ):
        """
        Args:
            num_examples: The number of examples to generate.
            max_length: The max_length of the dataset.
        """
        self.num_examples = num_examples
        self.tokenizer = tokenizer

    def __iter__(self):
        for _ in range(self.num_examples):
            input_ids = [self.tokenizer.bos_token_id]
            attention_mask = [1]
            token_type_ids = [1]
            drop = [False]
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "drop": drop,
            }


# endregion: Processors
################################################################################


################################################################################
# region: Collators


class BaseCollatorInput(TypedDict):
    """Dict with values that are lists of raw input_ids, attention_mask, and token_type_ids.

    The elements of the lists can be of different lengths.

    Attributes:
        input_ids (List[int]): The input ids.
        attention_mask (List[int]): The attention mask.
        token_type_ids (List[int]): The token type ids.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]


class DefaultCollator(Collator):
    """Simply stacks the input_ids, attention_mask, and token_type_ids and returns a batch."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> BaseBatch:
        return {
            "input_ids": torch.stack(
                [torch.tensor(e["input_ids"]) for e in examples]
            ),
            "attention_mask": torch.stack(
                [torch.tensor(e["attention_mask"]) for e in examples]
            ),
            "token_type_ids": torch.stack(
                [torch.tensor(e["token_type_ids"]) for e in examples]
            ),
        }


def pad_truncate(
    examples,
    max_len,
    pad_token_id,
    attn_extension: int = 0,
    type_extension: int = 2,
) -> BaseBatch:
    return {
        "input_ids": torch.tensor(
            [
                example["input_ids"][:max_len]
                + [pad_token_id] * max(0, max_len - len(example["input_ids"]))
                for example in examples
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                example["attention_mask"][:max_len]
                + [attn_extension]
                * max(0, max_len - len(example["attention_mask"]))
                for example in examples
            ],
            dtype=torch.long,
        ),
        "token_type_ids": torch.tensor(
            [
                example["token_type_ids"][:max_len]
                + [type_extension]
                * max(0, max_len - len(example["token_type_ids"]))
                for example in examples
            ],
            dtype=torch.long,
        ),
    }


def pad_prefix_suffix(tokenizer, examples, max_seq_len) -> BaseBatch:
    """
    [<varibale prefix>] [<bos> <variable suffix> <eos>]
    """
    prefixes = []
    suffixes = []
    max_prefix_len = 0
    max_suffix_len = 0
    for example in examples:
        bos_index = example["input_ids"].index(tokenizer.bos_token_id)
        prefix = example["input_ids"][: bos_index + 1]
        max_prefix_len = max(max_prefix_len, len(prefix))
        suffix = example["input_ids"][bos_index + 1 :]
        max_suffix_len = max(max_suffix_len, len(suffix))
        prefixes.append(prefix)
        suffixes.append(suffix)
    return {
        "input_ids": torch.tensor(
            [
                [tokenizer.pad_token_id]
                * max(0, max_prefix_len - len(prefixes[i]))
                + prefixes[i][:max_prefix_len]
                + suffixes[i][:max_suffix_len]
                + [tokenizer.pad_token_id]
                * max(0, max_suffix_len - len(suffixes[i]))
                for i in range(len(examples))
            ][:max_seq_len],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [0] * max(0, max_prefix_len - len(prefixes[i]))
                + [1] * len(prefixes[i])
                + [0] * len(suffixes[i])
                + [0] * max(0, max_suffix_len - len(suffixes[i]))
                for i in range(len(examples))
            ][:max_seq_len],
            dtype=torch.long,
        ).bool(),
        "token_type_ids": torch.tensor(
            [
                [2] * max(0, max_prefix_len - len(prefixes[i]))
                + [1] * len(prefixes[i])
                + [2] * len(suffixes[i])
                + [2] * max(0, max_suffix_len - len(suffixes[i]))
                for i in range(len(examples))
            ][:max_seq_len],
            dtype=torch.long,
        ),
    }


def pad_prefix_suffix2(
    examples,
    max_seq_len,
    pad_token_id,
    bos_token_id,
    eos_token_id,
    prefix_type_extension=0,
    suffix_type_extension=2,
    return_tensors=True,
) -> Union[BaseBatch, Dict[str, List[List[int]]]]:
    """
    [<varibale prefix>] [<bos> <variable suffix> <eos>]
    """
    prefixes = []
    suffixes = []
    max_prefix_len = 0
    max_suffix_len = 0
    for example in examples:
        bos_index = example["input_ids"].index(bos_token_id)
        prefix = example["input_ids"][: bos_index + 1]
        max_prefix_len = max(max_prefix_len, len(prefix))
        suffix = example["input_ids"][bos_index + 1 :]
        max_suffix_len = max(max_suffix_len, len(suffix))
        prefixes.append(prefix)
        suffixes.append(suffix)
    input_ids = [
        [pad_token_id] * max(0, max_prefix_len - len(prefixes[i]))
        + prefixes[i][:max_prefix_len]
        + suffixes[i][:max_suffix_len]
        + [pad_token_id] * max(0, max_suffix_len - len(suffixes[i]))
        for i in range(len(examples))
    ]
    attention_mask = [
        [0] * max(0, max_prefix_len - len(prefixes[i]))
        + [1] * len(prefixes[i])
        + [0] * len(suffixes[i])
        + [0] * max(0, max_suffix_len - len(suffixes[i]))
        for i in range(len(examples))
    ]
    token_type_ids = [
        [prefix_type_extension] * max(0, max_prefix_len - len(prefixes[i]))
        + [1] * len(prefixes[i])
        + [suffix_type_extension] * len(suffixes[i])
        + [suffix_type_extension] * max(0, max_suffix_len - len(suffixes[i]))
        for i in range(len(examples))
    ]
    if return_tensors:
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # type: ignore
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)  # type: ignore
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)  # type: ignore

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def pad_left_truncate(
    examples,
    max_len,
    pad_token_id,
    attn_extension: int = 0,
    type_extension: int = 2,
    return_tensors: bool = True,
) -> Union[BaseBatch, Dict[str, List[List[int]]]]:
    input_ids: List[List[int]] = [
        [pad_token_id] * max(0, max_len - len(example["input_ids"]))
        + example["input_ids"][:max_len]
        for example in examples
    ]
    attention_mask: List[List[int]] = [
        [attn_extension] * max(0, max_len - len(example["attention_mask"]))
        + example["attention_mask"][:max_len]
        for example in examples
    ]
    token_type_ids: List[List[int]] = [
        [type_extension] * max(0, max_len - len(example["token_type_ids"]))
        + example["token_type_ids"][:max_len]
        for example in examples
    ]
    if return_tensors:
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # type: ignore
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)  # type: ignore
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)  # type: ignore

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def pad_dynamic(
    examples,
    pad_token_id,
    attn_extension: int = 0,
    type_extension: int = 2,
) -> BaseBatch:
    max_len = max(len(e["input_ids"]) for e in examples)
    return pad_truncate(
        examples,
        max_len,
        pad_token_id,
        attn_extension=attn_extension,
        type_extension=type_extension,
    )


class DefaultCollatorWithPadding(DefaultCollator):
    """Like DefaultCollator, but pads (truncates if needed) the input_ids, attention_mask, and token_type_ids to self.max_length."""

    def get_max_len(self, examples: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(self, examples: List[BaseCollatorInput]) -> BaseBatch:
        max_len = self.get_max_len(examples)
        return pad_truncate(
            examples,
            max_len,
            self.tokenizer.pad_token_id,
            attn_extension=0,
            type_extension=2,
        )


class DefaultCollatorWithDynamicPadding(DefaultCollatorWithPadding):
    """Like DefaultCollator, but pads to the max length in the batch."""

    def get_max_len(self, examples: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in examples)
        return min(max_in_batch, self.block_size)


class DefaultMDLMCollator(Collator):
    """Adds MDLM noise to the input.

    Handles uneven lengths.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = True,
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        if loss_on_padding:
            # we are using attention_mask to compute NLL that ignores pads
            # So the model needs to know if the token is a pad or not
            # even if loss_on_padding is True
            self.attn_extension = 0
        else:
            self.attn_extension = 0

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        return self.noise_schedule.sample_t(batch_size)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> MDLMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        t: Float[TT, " *examples"] = self.sample_t(batch_size)
        noise_rate, total_noise = self.noise_schedule(t)
        base_batch = pad_truncate(
            examples,
            max_seq_len,
            self.tokenizer.pad_token_id,
            attn_extension=self.attn_extension,
            type_extension=2,
        )
        drop = torch.rand(
            (batch_size, max_seq_len), device=noise_rate.device
        ) < -(
            torch.expm1(-total_noise)[:, None]
        )  # shape: (batch_size, max_seq_len)
        if "token_type_ids" in base_batch:
            drop = torch.logical_and(base_batch["token_type_ids"] > 1, drop)
        base_batch["drop"] = drop  # type: ignore
        base_batch["t"] = t  # type: ignore
        base_batch["noise_rate"] = noise_rate  # type: ignore
        base_batch["total_noise"] = total_noise  # type: ignore
        base_batch["constraint"] = None  # type: ignore
        base_batch = cast(MDLMBatch, base_batch)
        return base_batch


class DefaultMDLMCollatorForPrediction(DefaultMDLMCollator):
    """ "Sets t=1 and masks all maskable tokens."""

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        return torch.ones((batch_size,))

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> MDLMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        t: Float[TT, " *examples"] = self.sample_t(batch_size)
        noise_rate, total_noise = self.noise_schedule(t)
        base_batch = pad_truncate(
            examples,
            max_seq_len,
            self.tokenizer.pad_token_id,
            attn_extension=self.attn_extension,
            type_extension=2,
        )
        # drop all tokens.
        drop = torch.ones((batch_size, max_seq_len), device=t.device)
        if "token_type_ids" in base_batch:
            drop = torch.logical_and(base_batch["token_type_ids"] > 1, drop)
        base_batch["drop"] = drop  # type: ignore
        base_batch["t"] = t  # type: ignore
        base_batch["noise_rate"] = noise_rate  # type: ignore
        base_batch["total_noise"] = total_noise  # type: ignore
        base_batch["constraint"] = (base_batch["token_type_ids"] == 1).long()
        base_batch = cast(MDLMBatch, base_batch)
        return base_batch


class DefaultIDLMCollator(Collator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        return_dense_target: bool = False,  # setting to two will increase the cpu memory usage
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.return_dense_target = return_dense_target

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        return self.noise_schedule.sample_t(batch_size)

    def sample_n_drops(self, total_noise) -> Integer[NA, " batch_size"]:
        return np.random.poisson(lam=total_noise)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> IDLMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        t: Float[TT, " *examples"] = self.sample_t(batch_size)
        noise_rate, total_noise = self.noise_schedule(t)
        n_drops = self.sample_n_drops(total_noise)
        drop_masks: List[Bool[TT, " max_seq_len"]] = []
        # other tensors
        input_ids: List[Integer[TT, " max_seq_len"]] = []
        attention_mask: List[Integer[TT, " max_seq_len"]] = []
        token_type_ids: List[Integer[TT, " max_seq_len"]] = []
        # Elements of the sparse tensor
        batch_indices: List[int] = []
        seq_indices: List[int] = []
        vocab_indices: List[int] = []
        values: List[int] = []
        for e, _example in enumerate(examples):
            example = {
                "input_ids": torch.tensor(
                    _example["input_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["input_ids"]))
                    * [self.tokenizer.pad_token_id]
                ),
                "attention_mask": torch.tensor(
                    _example["attention_mask"][:max_seq_len]
                    + (max_seq_len - len(_example["attention_mask"])) * [0]
                ).bool(),
                "token_type_ids": torch.tensor(
                    _example["token_type_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["token_type_ids"])) * [2]
                ),
            }
            non_pad = example["attention_mask"]
            input_ids.append(example["input_ids"])
            attention_mask.append(non_pad)
            token_type_ids.append(example["token_type_ids"])
            non_prefix = (
                example["token_type_ids"] > 1
                if "token_type_ids" in example
                else torch.ones_like(non_pad)
            )
            non_pad_and_non_prefix_indices = torch.logical_and(
                non_pad, non_prefix
            ).nonzero(as_tuple=True)[
                0
            ]  # shape: (*n_non_pad_and_non_prefix,)
            _seq_len = int(len(non_pad_and_non_prefix_indices))
            _n_drops = min(n_drops[e], _seq_len)
            _pre_drop_indices = (torch.randperm(_seq_len))[
                :_n_drops
            ]  # shape: (n_drops,)
            _drop_indices = non_pad_and_non_prefix_indices[
                _pre_drop_indices
            ]  # shape: (n_drops,)
            _drop_mask = torch.zeros((max_seq_len,), dtype=torch.bool)
            _drop_mask[_drop_indices] = True
            drop_masks.append(_drop_mask)
            # we maintain two indices:
            # 1. The index of the last non-dropped token: prev_remaining_j,
            #   initialized to the index of the token right before the first dropped token
            # 2. The current index: j
            # Check for empty tensor to handle the case blank input during prediction
            start = (
                int(non_pad_and_non_prefix_indices[0])
                if len(non_pad_and_non_prefix_indices) > 0
                else 0
            )
            prev_remaining_j = start - 1
            end = (
                int(non_pad_and_non_prefix_indices[-1]) + 1
                if len(non_pad_and_non_prefix_indices) > 0
                else 0
            )
            # Note: We could have used a loop over j going from 0 to max_seq_len,
            # but we don't need to, because we know for sure that there is
            # nothing to count before start and after end. So we save some time.
            # Note: This code should also work if the prefix is non-contiguous.
            for j in range(start, end):
                if _drop_mask[j]:
                    batch_indices.append(e)
                    seq_indices.append(prev_remaining_j)
                    vocab_indices.append(int(example["input_ids"][j].item()))
                    values.append(1)
                else:
                    prev_remaining_j = j

        target_ids = torch.sparse_coo_tensor(
            indices=[batch_indices, seq_indices, vocab_indices],  # type: ignore
            values=values,
            size=(batch_size, max_seq_len, self.vocab_size),
            check_invariants=False,
            is_coalesced=False,
        )
        return {
            "input_ids": collate_tensor_fn(input_ids),
            "attention_mask": collate_tensor_fn(attention_mask),
            "token_type_ids": collate_tensor_fn(token_type_ids),
            "drop": torch.stack(drop_masks, dim=0),
            "target_ids": (
                target_ids.to_dense()
                if self.return_dense_target
                else target_ids
            ),
            "t": t,
            "noise_rate": noise_rate,
            "total_noise": total_noise,
            "constraint": None,
        }


class DefaultILMCollator(Collator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = True,
        return_dense_target: bool = False,  # setting to two will increase the cpu memory usage
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.loss_on_padding = loss_on_padding
        self.return_dense_target = return_dense_target
        if self.loss_on_padding:
            self.attn_extension_id = 1
        else:
            self.attn_extension_id = 0

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_n_drops(self, seq_len: int) -> int:
        return np.random.randint(seq_len + 1)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ILMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        drop_masks: List[Bool[TT, " max_seq_len"]] = []
        # other tensors
        input_ids: List[Integer[TT, " max_seq_len"]] = []
        attention_mask: List[Integer[TT, " max_seq_len"]] = []
        token_type_ids: List[Integer[TT, " max_seq_len"]] = []
        # Elements of the sparse tensor
        batch_indices: List[int] = []
        seq_indices: List[int] = []
        vocab_indices: List[int] = []
        values: List[int] = []
        for e, _example in enumerate(examples):
            if len(_example["input_ids"]) > max_seq_len:
                raise ValueError(
                    f"Input ids length {len(_example['input_ids'])} is greater than max_seq_len {max_seq_len}"
                )
            example = {
                "input_ids": torch.tensor(
                    _example["input_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["input_ids"]))
                    * [self.tokenizer.pad_token_id]
                ),
                "attention_mask": torch.tensor(
                    _example["attention_mask"][:max_seq_len]
                    + (max_seq_len - len(_example["attention_mask"]))
                    * [self.attn_extension_id]
                ).bool(),
                "token_type_ids": torch.tensor(
                    _example["token_type_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["token_type_ids"])) * [2]
                ),
            }
            non_pad = example["attention_mask"]
            input_ids.append(example["input_ids"])
            attention_mask.append(non_pad)
            token_type_ids.append(example["token_type_ids"])
            non_prefix = (
                example["token_type_ids"] > 1
                if "token_type_ids" in example
                else torch.ones_like(non_pad)
            )
            non_pad_and_non_prefix_indices = torch.logical_and(
                non_pad, non_prefix
            ).nonzero(as_tuple=True)[
                0
            ]  # shape: (*n_non_pad_and_non_prefix,)
            _seq_len = int(len(non_pad_and_non_prefix_indices))
            _n_drops = self.sample_n_drops(_seq_len)
            _pre_drop_indices = (torch.randperm(_seq_len))[
                :_n_drops
            ]  # shape: (n_drops,)
            _drop_indices = non_pad_and_non_prefix_indices[
                _pre_drop_indices
            ]  # shape: (n_drops,)
            _drop_mask = torch.zeros((max_seq_len,), dtype=torch.bool)
            _drop_mask[_drop_indices] = True
            drop_masks.append(_drop_mask)
            # we maintain two indices:
            # 1. The index of the last non-dropped token: prev_remaining_j,
            #   initialized to the index of the token right before the first dropped token
            # 2. The current index: j
            # Check for empty tensor to handle the case blank input during prediction
            start = (
                int(non_pad_and_non_prefix_indices[0])
                if len(non_pad_and_non_prefix_indices) > 0
                else 0
            )
            prev_remaining_j = start - 1
            end = (
                int(non_pad_and_non_prefix_indices[-1]) + 1
                if len(non_pad_and_non_prefix_indices) > 0
                else 0
            )
            # Note: We could have used a loop over j going from 0 to max_seq_len,
            # but we don't need to, because we know for sure that there is
            # nothing to count before start and after end. So we save some time.
            # Note: This code should also work if the prefix is non-contiguous.
            for j in range(start, end):
                if _drop_mask[j]:
                    batch_indices.append(e)
                    seq_indices.append(prev_remaining_j)
                    vocab_indices.append(int(example["input_ids"][j].item()))
                    values.append(1)
                else:
                    prev_remaining_j = j

        target_ids = torch.sparse_coo_tensor(
            indices=[batch_indices, seq_indices, vocab_indices],  # type: ignore
            values=values,
            size=(batch_size, max_seq_len, self.vocab_size),
            check_invariants=False,
            is_coalesced=False,
        )
        return {
            "input_ids": collate_tensor_fn(input_ids),
            "attention_mask": collate_tensor_fn(attention_mask),
            "token_type_ids": collate_tensor_fn(token_type_ids),
            "drop": torch.stack(drop_masks, dim=0),
            "target_ids": (
                target_ids.to_dense()
                if self.return_dense_target
                else target_ids
            ),
            "constraint": None,
        }


class DefaultITCollator(Collator):
    """Same as DefaultILMCollator but also adds counts of each slo"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = True,
        return_dense_target: bool = False,  # setting to two will increase the cpu memory usage
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.loss_on_padding = loss_on_padding
        self.return_dense_target = return_dense_target
        if self.loss_on_padding:
            self.attn_extension_id = 1
        else:
            self.attn_extension_id = 0

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_n_drops(self, seq_len: int) -> int:
        return np.random.randint(seq_len + 1)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ITBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        drop_masks: List[Bool[TT, " max_seq_len"]] = []
        counts: List[Integer[TT, " max_seq_len"]] = []
        # other tensors
        input_ids: List[Integer[TT, " max_seq_len"]] = []
        attention_mask: List[Integer[TT, " max_seq_len"]] = []
        token_type_ids: List[Integer[TT, " max_seq_len"]] = []
        # Elements of the sparse tensor
        batch_indices: List[int] = []
        seq_indices: List[int] = []
        vocab_indices: List[int] = []
        values: List[int] = []
        for e, _example in enumerate(examples):
            if len(_example["input_ids"]) > max_seq_len:
                raise ValueError(
                    f"Input ids length {len(_example['input_ids'])} is greater than max_seq_len {max_seq_len}"
                )
            example = {
                "input_ids": torch.tensor(
                    _example["input_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["input_ids"]))
                    * [self.tokenizer.pad_token_id]
                ),
                "attention_mask": torch.tensor(
                    _example["attention_mask"][:max_seq_len]
                    + (max_seq_len - len(_example["attention_mask"]))
                    * [self.attn_extension_id]
                ).bool(),
                "token_type_ids": torch.tensor(
                    _example["token_type_ids"][:max_seq_len]
                    + (max_seq_len - len(_example["token_type_ids"])) * [2]
                ),
            }
            non_pad = example["attention_mask"]
            input_ids.append(example["input_ids"])
            attention_mask.append(non_pad)
            token_type_ids.append(example["token_type_ids"])
            non_prefix = (
                example["token_type_ids"] > 1
                if "token_type_ids" in example
                else torch.ones_like(non_pad)
            )
            non_pad_and_non_prefix_indices = torch.logical_and(
                non_pad, non_prefix
            ).nonzero(as_tuple=True)[
                0
            ]  # shape: (*n_non_pad_and_non_prefix,)
            _seq_len = int(len(non_pad_and_non_prefix_indices))
            _n_drops = self.sample_n_drops(_seq_len)
            _pre_drop_indices = (torch.randperm(_seq_len))[
                :_n_drops
            ]  # shape: (n_drops,)
            _drop_indices = non_pad_and_non_prefix_indices[
                _pre_drop_indices
            ]  # shape: (n_drops,)
            _drop_mask = torch.zeros((max_seq_len,), dtype=torch.bool)
            _drop_mask[_drop_indices] = True
            drop_masks.append(_drop_mask)
            # we maintain two indices:
            # 1. The index of the last non-dropped token: prev_remaining_j,
            #   initialized to the index of the token right before the first dropped token
            # 2. The current index: j
            # Check for empty tensor to handle the case blank input during prediction
            # start = (
            #    int(non_pad_and_non_prefix_indices[0])
            #    if len(non_pad_and_non_prefix_indices) > 0
            #    else 0
            # )
            # prev_remaining_j = start - 1
            # end = (
            #    int(non_pad_and_non_prefix_indices[-1]) + 1
            #    if len(non_pad_and_non_prefix_indices) > 0
            #    else 0
            # )
            # Note: We could have used a loop over j going from 0 to max_seq_len,
            # but we don't need to, because we know for sure that there is
            # nothing to count before start and after end. So we save some time.
            # Note: This code should also work if the prefix is non-contiguous.
            start = 1
            end = max_seq_len
            prev_remaining_j = start - 1
            _counts = torch.zeros_like(_drop_mask, dtype=torch.long)
            counts.append(_counts)

            for j in range(start, end):
                if _drop_mask[j]:
                    batch_indices.append(e)
                    seq_indices.append(prev_remaining_j)
                    vocab_indices.append(int(example["input_ids"][j].item()))
                    values.append(1)
                    _counts[prev_remaining_j] += 1
                else:
                    # this is speicial of ITCollator
                    batch_indices.append(e)
                    seq_indices.append(j - 1)
                    vocab_indices.append(self.tokenizer.eos_token_id)
                    values.append(1)
                    prev_remaining_j = j

        target_ids = torch.sparse_coo_tensor(
            indices=[batch_indices, seq_indices, vocab_indices],  # type: ignore
            values=values,
            size=(batch_size, max_seq_len, self.vocab_size),
            check_invariants=False,
            is_coalesced=False,
        )
        return {
            "input_ids": collate_tensor_fn(input_ids),
            "attention_mask": collate_tensor_fn(attention_mask),
            "token_type_ids": collate_tensor_fn(token_type_ids),
            "drop": torch.stack(drop_masks, dim=0),
            "target_ids": (
                target_ids.to_dense()
                if self.return_dense_target
                else target_ids
            ),
            "constraint": None,
            "counts": torch.stack(counts, dim=0),
        }


class DefaultILMWithLengthClassificationCollator(DefaultILMCollator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = False,
        return_dense_target: bool = False,  # setting to two will increase the cpu memory usage
    ):
        if loss_on_padding:
            logger.warning(
                "loss_on_padding is true in collator for length classification setting."
                " This is not the typical setting for ILMWithLengthClassification. "
            )
        super().__init__(
            tokenizer,
            block_size,
            noise_schedule,
            loss_on_padding=loss_on_padding,
            return_dense_target=return_dense_target,
        )


class DefaultILMWithLengthClassificationCollatorForWrappedPaddedSequences(
    DefaultILMWithLengthClassificationCollator
):
    """For pre-padded sequences, that are potentially also wrapped/grouped into blocks.

    Note:
       1. The collator prepares `constraint` tensor, where 1 is set for CLS tokens.
    """

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ILMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        drop_masks: List[Bool[TT, " max_seq_len"]] = []
        # other tensors
        input_ids = collections.deque()
        attention_mask = collections.deque()
        token_type_ids = collections.deque()
        constraint = collections.deque()
        # Elements of the sparse tensor
        batch_indices: List[int] = []
        seq_indices: List[int] = []
        vocab_indices: List[int] = []
        values: List[int] = []
        for e, _example in enumerate(examples):
            non_pad = _example["attention_mask"]
            input_ids.append(_example["input_ids"])
            attention_mask.append(_example["attention_mask"])
            token_type_ids.append(_example["token_type_ids"])
            # _constraint = []  # length: (max_seq_len,)
            non_pad_and_non_cls_non_bos_indices = (
                []
            )  # length: (num_non_pad_and_non_cls,)
            for _i, (_not_pad, _type_id) in enumerate(
                zip(non_pad, _example["token_type_ids"])
            ):
                if _type_id > 1:
                    if _not_pad or self.loss_on_padding:
                        non_pad_and_non_cls_non_bos_indices.append(_i)
                    # _constraint.append(False)
                else:  # _type_id == 0:
                    # _constraint.append(
                    #    True
                    # )  # we don't want token prediction loss on CLS
                    pass
            # constraint.append(_constraint)
            _n_drops = self.sample_n_drops(
                len(non_pad_and_non_cls_non_bos_indices)
            )
            _drop_indices = np.random.default_rng().choice(
                non_pad_and_non_cls_non_bos_indices,
                size=_n_drops,
                replace=False,
            )  # length: (n_drops,)
            _drop_mask = torch.zeros((max_seq_len,), dtype=torch.bool)
            _drop_mask[_drop_indices] = True
            drop_masks.append(_drop_mask)
            # we maintain two indices:
            # 1. The index of the last non-dropped token: prev_remaining_j,
            #   initialized to the index of the token right before the first dropped token
            # 2. The current index: j
            # Check for empty tensor to handle the case blank input during prediction
            start = (
                int(non_pad_and_non_cls_non_bos_indices[0])
                if len(non_pad_and_non_cls_non_bos_indices) > 0
                else 0
            )
            prev_remaining_j = max(
                0, start - 1
            )  # -1 because we want to start at bos
            end = (
                int(non_pad_and_non_cls_non_bos_indices[-1]) + 1
                if len(non_pad_and_non_cls_non_bos_indices) > 0
                else 0
            )
            # Note: We could have used a loop over j going from 0 to max_seq_len,
            # but we don't need to, because we know for sure that there is
            # nothing to count before start and after end. So we save some time.
            # Note: This code should also work if the prefix is non-contiguous.

            # We need to reset when type_id=0 is encountered
            for j in range(start, end):
                # skip if type_id=0, i.e. CLS, assuming that it will be followed by BOS
                if _example["token_type_ids"][j] == 0:
                    prev_remaining_j = (
                        j + 1
                    )  # just to be safe set prev_remaining_j to the next token which is expected to be BOS
                    continue
                if _drop_mask[j]:
                    batch_indices.append(e)
                    seq_indices.append(prev_remaining_j)
                    vocab_indices.append(_example["input_ids"][j])
                    values.append(1)
                else:
                    prev_remaining_j = j

        target_ids = torch.sparse_coo_tensor(
            indices=[batch_indices, seq_indices, vocab_indices],  # type: ignore
            values=values,
            size=(batch_size, max_seq_len, self.vocab_size),
            check_invariants=False,
            is_coalesced=False,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "drop": torch.stack(drop_masks, dim=0),
            "target_ids": (
                target_ids.to_dense()
                if self.return_dense_target
                else target_ids
            ),
            # "constraint": torch.tensor(constraint, dtype=torch.bool),
            "constraint": None,
        }


class DefaultIDLMCollatorWithDynamicPadding(DefaultIDLMCollator):
    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in batch)
        return min(max_in_batch, self.block_size)


class DefaultIDLMCollatorForPrediction(DefaultIDLMCollator):
    """Drop all dropable tokens and set t=1."""

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in batch)
        return min(max_in_batch, self.block_size)

    def sample_n_drops(self, total_noise) -> Integer[NA, " batch_size"]:
        batch_size = len(total_noise)
        return np.full((batch_size,), fill_value=self.block_size)

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        return torch.ones((batch_size,))


class DefaultILMCollatorForPrediction(DefaultILMCollator):
    """Drop all dropable tokens."""

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in batch)
        return min(max_in_batch, self.block_size)

    def sample_n_drops(self, seq_len: int) -> int:
        return seq_len


class DefaultITCollatorForPrediction(DefaultITCollator):
    """Drop all dropable tokens and set t=1."""

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in batch)
        return min(max_in_batch, self.block_size)

    def sample_n_drops(self, seq_len: int) -> int:
        return seq_len


class DefaultILMWithLengthClassificationCollatorForPrediction(
    DefaultILMWithLengthClassificationCollator
):
    """Drop all dropable tokens"""

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in batch)
        return min(max_in_batch, self.block_size)

    def sample_n_drops(self, seq_len: int) -> int:
        return seq_len


# ----------------------------------
class DefaultARLMCollator(Collator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ARLMBatch:
        """
        constraint - set 1 for tokens to not be predicted (prefix + pad)
        attention mask - set 1 for prefix only
        drop - for masking whatever we don't predict (also used for masking arlm loss)
        """
        max_seq_len = self.get_max_len(examples)
        pad_token_id = self.tokenizer.pad_token_id
        batch = pad_truncate(examples, max_seq_len, pad_token_id)
        constraint = (
            (batch["token_type_ids"] == 1)
            | (batch["input_ids"] == pad_token_id)
        ).long()
        drop = (
            (batch["token_type_ids"] == 1)
            | (batch["input_ids"] == pad_token_id)
        ).long()
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "drop": drop,
            "constraint": constraint,
        }


class DefaultARLMCollatorForPrediction(DefaultARLMCollator):
    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in batch)
        return min(max_in_batch, self.block_size)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ARLMBatch:
        max_seq_len = self.get_max_len(examples)
        base_batch = pad_truncate(
            examples, max_seq_len, self.tokenizer.pad_token_id
        )
        base_batch["constraint"] = (
            (base_batch["token_type_ids"] == 1)
            | (base_batch["input_ids"] == self.tokenizer.pad_token_id)
        ).long()
        base_batch["attention_mask"] = (
            (base_batch["token_type_ids"] == 1)
            & (base_batch["input_ids"] != self.tokenizer.pad_token_id)
        ).bool()
        base_batch["drop"] = (
            (base_batch["token_type_ids"] == 2)
            | (base_batch["input_ids"] == self.tokenizer.pad_token_id)
        ).long()
        base_batch = cast(ARLMBatch, base_batch)
        return base_batch

class DefaultMLMCollator(Collator):
    """Adds MLM noise to the input.

    Handles uneven lengths.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
        loss_on_padding: bool = True,
    ):
        self.block_size = block_size
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        if loss_on_padding:
            # we are using attention_mask to compute NLL that ignores pads
            # So the model needs to know if the token is a pad or not
            # even if loss_on_padding is True
            self.attn_extension = 0
        else:
            self.attn_extension = 0

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        return self.noise_schedule.sample_t(batch_size)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> MLMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        t: Float[TT, " *examples"] = self.sample_t(batch_size)
        noise_rate, total_noise = self.noise_schedule(t)
        base_batch = pad_truncate(
            examples,
            max_seq_len,
            self.tokenizer.pad_token_id,
            attn_extension=self.attn_extension,
            type_extension=2,
        )
        drop = torch.rand(
            (batch_size, max_seq_len), device=noise_rate.device
        ) < -(
            torch.expm1(-total_noise)[:, None]
        )  # shape: (batch_size, max_seq_len)
        if "token_type_ids" in base_batch:
            drop = torch.logical_and(base_batch["token_type_ids"] > 1, drop)
        base_batch["drop"] = drop  # type: ignore
        base_batch["t"] = t  # type: ignore
        base_batch["noise_rate"] = noise_rate  # type: ignore
        base_batch["total_noise"] = total_noise  # type: ignore
        base_batch["constraint"] = None  # type: ignore
        base_batch = cast(MLMBatch, base_batch)
        return base_batch


class DefaultMLMCollatorForPrediction(DefaultMLMCollator):
    """ "Sets t=1 and masks all maskable tokens."""

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def sample_t(self, batch_size: int) -> Float[TT, " batch_size"]:
        return torch.ones((batch_size,))

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> MDLMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        t: Float[TT, " *examples"] = self.sample_t(batch_size)
        noise_rate, total_noise = self.noise_schedule(t)
        base_batch = pad_truncate(
            examples,
            max_seq_len,
            self.tokenizer.pad_token_id,
            attn_extension=self.attn_extension,
            type_extension=2,
        )
        # drop all tokens.
        drop = torch.ones((batch_size, max_seq_len), device=t.device)
        if "token_type_ids" in base_batch:
            drop = torch.logical_and(base_batch["token_type_ids"] > 1, drop)
        base_batch["drop"] = drop  # type: ignore
        base_batch["t"] = t  # type: ignore
        base_batch["noise_rate"] = noise_rate  # type: ignore
        base_batch["total_noise"] = total_noise  # type: ignore
        base_batch["constraint"] = (base_batch["token_type_ids"] == 1).long()
        base_batch = cast(MDLMBatch, base_batch)
        return base_batch

# endregion: Collators
################################################################################


################################################################################
# region: Utilities


def print_batch_base(
    datamodule: BaseDataModule,
    batch: Dict[str, Any],
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
        print("token_type_ids:")
        print(batch["token_type_ids"][0])


def print_batch_idlm(
    datamodule: BaseDataModule,
    batch: IDLMBatch,
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
        print("token_type_ids:")
        print(batch["token_type_ids"][0])
        print("drop:")
        print(batch["drop"][0])
        print("target_ids:")
        print(batch["target_ids"][0].to_sparse())
        print("t:")
        print(batch["t"][0])
        print("noise_rate:")
        print(batch["noise_rate"][0])
        print("total_noise:")
        print(batch["total_noise"][0])
        print("constraint:")
        print(
            batch["constraint"][0] if batch["constraint"] is not None else None
        )


def print_batch_mdlm(
    datamodule: BaseDataModule,
    batch: MDLMBatch,
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
        print("token_type_ids:")
        print(
            batch["token_type_ids"][0]
            if batch["token_type_ids"] is not None
            else None
        )
        print("drop:")
        print(batch["drop"][0])
        print("t:")
        print(batch["t"][0])
        print("noise_rate:")
        print(batch["noise_rate"][0])
        print("total_noise:")
        print(batch["total_noise"][0])
        print("constraint:")
        print(
            batch["constraint"][0] if batch["constraint"] is not None else None
        )


def print_batch_ilm(
    datamodule: BaseDataModule,
    batch: ILMBatch,
    split: Literal["train", "val", "test", "predict"],
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
                else (
                    self.test_dataloader_names.get(dataloader_idx, "")
                    if split == "test"
                    else (
                        self.predict_dataloader_names.get(dataloader_idx, "")
                        if split == "predict"
                        else self.predict_dataloader_names.get(
                            dataloader_idx, ""
                        )
                    )
                )
            )
        )
    if split in ["train", "val", "test", "predict"]:
        logger.info(
            f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
        )
        print("input tokens:")
        print(self.tokenizer.decode(batch["input_ids"][0]))
        print("input_ids:")
        print(batch["input_ids"][0])
        print("attention_mask:")
        print(batch["attention_mask"][0])
        print("token_type_ids:")
        print(batch["token_type_ids"][0])
        print("drop:")
        print(batch["drop"][0])
        print("target_ids:")
        print(
            batch["target_ids"][0].to_sparse()
            if batch is not None and batch.get("target_ids", None) is not None
            else None
        )
        print("constraint:")
        print(
            batch["constraint"][0] if batch["constraint"] is not None else None
        )


def print_batch_it(
    datamodule: BaseDataModule,
    batch: ITBatch,
    split: Literal["train", "val", "test", "predict"],
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
                else (
                    self.test_dataloader_names.get(dataloader_idx, "")
                    if split == "test"
                    else (
                        self.predict_dataloader_names.get(dataloader_idx, "")
                        if split == "predict"
                        else self.predict_dataloader_names.get(
                            dataloader_idx, ""
                        )
                    )
                )
            )
        )
    if split in ["train", "val", "test", "predict"]:
        logger.info(
            f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
        )
        print("input tokens:")
        print(self.tokenizer.decode(batch["input_ids"][0]))
        print("input_ids:")
        print(batch["input_ids"][0])
        print("attention_mask:")
        print(batch["attention_mask"][0])
        print("token_type_ids:")
        print(batch["token_type_ids"][0])
        print("drop:")
        print(batch["drop"][0])
        print("counts:")
        print(batch["counts"][0])
        print("target_ids:")
        print(
            batch["target_ids"][0].to_sparse()
            if batch is not None and batch.get("target_ids", None) is not None
            else None
        )
        print("constraint:")
        print(
            batch["constraint"][0] if batch["constraint"] is not None else None
        )


def print_batch_arlm(
    datamodule: BaseDataModule,
    batch: ILMBatch,
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
        print("token_type_ids:")
        print(batch["token_type_ids"][0])
        print("drop:")
        print(batch["drop"][0])
        print("constraint:")
        print(
            batch["constraint"][0] if batch["constraint"] is not None else None
        )

def print_batch_mlm(
    datamodule: BaseDataModule,
    batch: MLMBatch,
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
        print("token_type_ids:")
        print(batch["token_type_ids"][0])
        print("drop:")
        print(batch["drop"][0])
        print("constraint:")
        print(
            batch["constraint"][0] if batch["constraint"] is not None else None
        )


class PrintBatchCallback(L.Callback):

    @rank_zero_only
    def on_fit_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # obtain a fresh dl just for printing
        train_dl = trainer.datamodule.train_dataloader()
        if not hasattr(trainer.datamodule, "print_batch"):
            return
        logger.info(f"Printing batch for train dataloader")
        batch = next(iter(train_dl))
        trainer.datamodule.print_batch(batch, "train")

    def on_sanity_check_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # obtain a fresh dl just for printing
        val_dl = trainer.datamodule.val_dataloader()
        if val_dl is None:
            return
        val_dls = val_dl if isinstance(val_dl, list) else [val_dl]
        if not hasattr(trainer.datamodule, "print_batch"):
            return
        logger.info("Printing batch for val dataloaders")
        for i, val_dl in enumerate(val_dls):
            batch = next(iter(val_dl))
            trainer.datamodule.print_batch(batch, "val", i)


# endregion: Utilities
################################################################################
