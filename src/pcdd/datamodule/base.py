from copy import deepcopy
import itertools
import logging
from pathlib import Path
import re
from traceback import print_tb
from typing import (
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
    Sequence,
    Dict,
    Any,
    TypeVar,
    cast,
)
from jaxtyping import Integer, Bool
import numpy as np
from torch import Tensor as TT
from more_itertools import flatten, chunked
import datasets
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, processors
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from lightning import Callback, LightningDataModule
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from pcdd.utils.os import get_num_processes
from pcdd.utils.rank_zero import rank_zero_only, RankedLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers import AutoTokenizer

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


# region: Types


class HFExample(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]


class TensorHFExampleType(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor


HFPreBatch = List[HFExample]
TensorHFPreBatch = List[TensorHFExampleType]


class HFBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class HFDatasetRequiredKwargs(TypedDict):
    path: str


class HFDatasetOptionalKwargs(TypedDict, total=False):
    name: Optional[str]
    data_dir: Optional[str]
    data_files: Optional[
        Union[str, Sequence[str], Dict[str, Union[str, Sequence[str]]]]
    ]
    split: Optional[str]
    cache_dir: Optional[str]
    keep_in_memory: Optional[bool]
    token: Optional[Union[bool, str]]
    streaming: bool
    num_proc: Optional[int]


class HFDatasetKwargs(HFDatasetRequiredKwargs, HFDatasetOptionalKwargs):
    pass


DatasetT = Union[
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict,
]


class DataLoaderKwargs(TypedDict):
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool


# endregion: Types

# region: Custom Datasets


def get_repeated_chars_dataset(
    dataset_name: str,
    kwargs: Optional[Dict[str, Any]],
    split: Optional[str],
) -> datasets.Dataset:
    # only called when a fresh dataset is requested (no cache available)
    if split is None:
        raise ValueError("Split is required for repeated chars dataset")
    rnd_generator = np.random.default_rng(123)
    train_size = 10000
    # fmt: off
    lengths = {"a": 1,"b":1,"c":1,"d":1,"e":5,"f":5,"g":5,"h":5,"i":5,"j":10,"k":10,"l":10,"m":10,"n":10,"o":15,"p":15,"q":15,"r":15,"s":15,"t":20,"u":20,"v":20,"w":20,"x":20,"y":25,"z":25} # noqa
    strings = {ch: ch * lengths[ch] for ch in lengths}
    # fmt: on
    chars = list(lengths.keys())
    if split == "train":
        char_generator = rnd_generator.choice(
            chars, size=train_size, replace=True
        )
    else:
        char_generator = list(chars)

    def generate_example():
        for char in char_generator:
            yield {"text": strings[char]}

    dataset = datasets.Dataset.from_generator(
        generate_example,
        features=datasets.Features({"text": datasets.Value("string")}),
    )
    dataset = cast(datasets.Dataset, dataset)
    return dataset


LOCAL_DATASETS: Dict[
    str, Callable[[str, Optional[Dict[str, Any]], Optional[str]], DatasetT]
] = {
    "repeated_chars": get_repeated_chars_dataset,
}
# endregion

# region: Tokenizer


# region: LowerCaseCharTokenizer
# Adapted from Text8Tokenizer in MDLM.
class LowerCaseCharTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()
        }
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int["[UNK]"]
        )

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Adds special tokens like BOS/EOS based on the model's requirements."""
        if token_ids_1 is not None:
            raise ValueError(
                "token_ids_1 is not supported for LowerCaseCharTokenizer"
            )
        token_ids = token_ids_0
        bos_id = self._vocab_str_to_int.get(self.bos_token, None)
        eos_id = self._vocab_str_to_int.get(self.eos_token, None)
        # DEBUG: no eos
        eos_id = None

        if bos_id is not None and eos_id is not None:
            return [bos_id] + token_ids + [eos_id]
        elif bos_id is not None:
            return [bos_id] + token_ids
        elif eos_id is not None:
            return token_ids + [eos_id]
        return token_ids


# endregion: LowerCaseCharTokenizer


def create_pre_trained_tokenizer(
    tokenizer: PreTrainedTokenizer,
    bos_token: Optional[str] = None,
    eos_token: Optional[str] = None,
    pad_token: Optional[str] = None,
    mask_token: Optional[str] = None,
    unk_token: Optional[str] = None,
):
    """Add special tokens to a pretrained tokenizer."""
    add_special_token(tokenizer, bos_token, "bos_token")
    add_special_token(tokenizer, eos_token, "eos_token")
    add_special_token(tokenizer, pad_token, "pad_token")
    add_special_token(tokenizer, mask_token, "mask_token")
    add_special_token(tokenizer, unk_token, "unk_token")
    return tokenizer


def create_tokenizer_from_file(
    tokenizer_path: str,
    bos_token: Optional[str] = None,
    eos_token: Optional[str] = None,
    pad_token: Optional[str] = None,
    mask_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    add_eos_post_processor: bool = True,
) -> PreTrainedTokenizerFast:
    """Create a tokenizers.Tokenizer from a file."""
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    if add_eos_post_processor:
        if eos_token is None:
            logger.warning(
                "add_eos_post_processor is True but eos_token is None. This will not add EOS post processor."
            )
        else:
            _tokenizer.post_processor = create_eos_post_processor(
                eos_token, _tokenizer.token_to_id(eos_token)
            )
    # ref: https://github.com/huggingface/tokenizers/issues/777
    # TODO: Set block size for the model and the tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_tokenizer,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        unk_token=unk_token,
        bos_token=bos_token,
    )
    return tokenizer


def create_eos_post_processor(eos_token: str, eos_token_id: int):
    """Create a post processor for the tokenizer to add EOS token to the end of the sequence.

    See https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/pipeline#postprocessing and
    https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/api/post-processors#tokenizers.processors.TemplateProcessing
    for more details.
    """
    post_processor = processors.TemplateProcessing(
        single=f"$0 {eos_token}",
        pair=f"$A:0 $B:1 {eos_token}",
        special_tokens=[(eos_token, eos_token_id)],
    )
    return post_processor


def create_bert_tokenizer(
    pretrained_model_name_or_path: str = "bert-base-uncased",
):
    """BERT based tokenizer.
    Adds BOS=CLS and EOS=SEP to bert tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    # BERT does not have a BOS and EOS
    # It has tokenizer.all_special_tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    tokenizer.bos_token = tokenizer.cls_token  # from mdlm
    tokenizer.eos_token = tokenizer.sep_token  # from mdlm
    return tokenizer


def add_special_token(
    tokenizer: PreTrainedTokenizer,
    special_token: Optional[str],
    attribute: str,
):
    if special_token is None:
        return tokenizer
    if getattr(tokenizer, attribute) is None:
        raise ValueError(
            f"{tokenizer.__class__.__name__}.{attribute} is not present in the tokenizer."
        )
    if getattr(tokenizer, attribute) != special_token:
        rank_zero_warn(
            f"Old {tokenizer.__class__.__name__}.{attribute}: {getattr(tokenizer, attribute)} not the same as {special_token}. Overwriting."
        )
    tokenizer.add_special_tokens({attribute: special_token})
    rank_zero_info(f"tokenizer.{attribute}: {getattr(tokenizer, attribute)}")


# endregion: Tokenizer


# region: Collation functions
def create_collate_fn_for_hf_data_using_pretrained_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    return_tensors="pt",
    padding=True,
    **kwargs,
):
    """
    Args:
        kwargs:  See https://github.com/huggingface/transformers/blob/3a8eb74668e9c2cc563b2f5c62fac174797063e0/src/transformers/tokenization_utils_base.py#L3184
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                    Select a strategy to pad the returned sequences (according to the model's padding side and padding
                    index) among:

                    - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                    sequence if provided).
                    - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                    acceptable input length for the model if that argument is not provided.
                    - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                    lengths).
                max_length (`int`, *optional*):
                    Maximum length of the returned list and optionally padding length (see above).
                pad_to_multiple_of (`int`, *optional*):
                    If set will pad the sequence to a multiple of the provided value.

                    This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                    `>= 7.5` (Volta).
                padding_side (`str`, *optional*):
                    The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                    Default value is picked from the class attribute of the same name.
                return_attention_mask (`bool`, *optional*):
                    Whether to return the attention mask. If left to the default, will return the attention mask according
                    to the specific tokenizer's default, defined by the `return_outputs` attribute.

                    [What are attention masks?](../glossary#attention-mask)
                return_tensors (`str` or [`~utils.TensorType`], *optional*):
                    If set, will return tensors instead of list of python integers. Acceptable values are:

                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return Numpy `np.ndarray` objects.
                verbose (`bool`, *optional*, defaults to `True`):
                    Whether or not to print more information and warnings.

    Notes: If using `padding='max_length` and leaving `max_length=None`, then the tokenizer should have attribute `model_max_length` to use.
    """

    # make sure that the tokenizer has pad token
    def collate_fn(batch: HFPreBatch):
        return pad_without_fast_tokenizer_warning(
            tokenizer,
            batch,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )

    return collate_fn


def group_texts(
    examples: Dict[str, List[List[int]]], block_size: int, bos: int, eos: int
) -> Dict[str, List[List[int]]]:
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    # TODO (padding): look into not dropping the remainder but rather padding it.
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of
    # this drop, you can customize this part to your needs.
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append(
            [bos] + concatenated_examples[i : i + new_block_size] + [eos]
        )
        _attn_masks.append(torch.ones(block_size))
    result["input_ids"] = _values
    result["attention_mask"] = _attn_masks
    return result


# endregion


# region: Detokenizers
# MAYBE: Fix the detokenizers
def lm1b_detokenizer(x):
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(
        r" \'(\w+)", r"'\1", x
    )  # remove extra space before single quotes like "they 've"
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


DETOKENIZERS = {
    "lm1b": lm1b_detokenizer,
}
# endregion


# region: Debugging
@rank_zero_only
def print_batch(
    dataloader: torch.utils.data.DataLoader,
    name: str,
    tokenizer: PreTrainedTokenizer,
    num_tokens_to_print: int = 128,
) -> None:
    batch = next(iter(dataloader))
    logger.info(f"Printing batch for {name}")
    # expect the batch to be a dict if not raise an error
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{name}.{k}: {v.shape} {v.dtype}")
        else:
            print(f"{name}.{k}: {type(v)}")
    seq_len = min(num_tokens_to_print, batch["input_ids"].shape[1])
    first = tokenizer.decode(
        batch["input_ids"][0, :seq_len], skip_special_tokens=True
    )
    last = tokenizer.decode(
        batch["input_ids"][0, 0 - seq_len :], skip_special_tokens=True
    )
    print(f"{name} initial tokens: {first}")
    print(f"{name} initial ids: {batch['input_ids'][0, :seq_len]}")
    print(f"{name} last tokens: {last}")
    print(f"{name} last ids: {batch['input_ids'][0, -seq_len:]}")


class PrintBatchCallback(Callback):
    def __init__(self, num_tokens_to_print: int = 64):
        self.num_tokens_to_print = num_tokens_to_print

    def get_print_fn(self, trainer: "Trainer") -> Callable:
        has_print_fn = hasattr(trainer.datamodule, "print_batch")
        if has_print_fn:
            print_fn = trainer.datamodule.print_batch
        else:
            print_fn = print_batch
        return print_fn

    def on_fit_start(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        # obtain a fresh dl just for printing
        train_dl = trainer.datamodule.train_dataloader()
        print_fn = self.get_print_fn(trainer)
        print_fn(
            train_dl,
            "train",
            trainer.datamodule.tokenizer,
            self.num_tokens_to_print,
        )

    def on_sanity_check_start(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        # obtain a fresh dl just for printing
        val_dl = trainer.datamodule.val_dataloader()
        print_fn = self.get_print_fn(trainer)
        print_fn(
            val_dl,
            "val",
            trainer.datamodule.tokenizer,
            self.num_tokens_to_print,
        )


# endregion: Debugging


# region: DataModule
class DiffusionForTextDataModule(LightningDataModule):
    """
    Collation Strategies:

    1. Wrap:
        - [BOS] + seq1 + [EOS] + seq2_part1 + [EOS] | block_size
        - [BOS] + seq2_part2 + [EOS] + seq3_part1 + [EOS] | block_size
        ...
        - [BOS] + .... + [EOS] + [PAD] ... | block_size
    2. Pad:
        - seq1 + [EOS] + [PAD] ... | max_length_in_batch
    3. pad_truncate:
        - seq1 + [EOS] + [PAD] ... | fixed_max_length
    """

    def __init__(
        self,
        manual_cache_dir: str,
        dataset_name: str,
        hf_dataset_kwargs: HFDatasetKwargs,
        train_dataloader_kwargs: DataLoaderKwargs,
        val_dataloader_kwargs: DataLoaderKwargs,
        test_dataloader_kwargs: Optional[DataLoaderKwargs],
        tokenizer: PreTrainedTokenizer,
        global_batch_size: int,
        block_size: int = 1024,
        collation_strategy: Literal["wrap", "pad", "pad_truncate"] = "wrap",
        train_split: Optional[str] = "train",
        val_split: Optional[str] = "validation",
        test_split: Optional[str] = "test",
        text_field_name: str = "text",
        detokenizer: Optional[Literal["lm1b"]] = None,
        num_dataset_workers: Optional[int] = None,
        columns_to_remove: Optional[List[str]] = None,
        rewrite_manual_cache: bool = False,
        local_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            manual_cache_dir: We do manual caching using this directory. When this is specified the pipeline in
                `prepare_data` will be skipped as we will load using `datasets.load_from_disk`.
            hf_dataset_kwargs: The keyword arguments for the Hugging Face dataset. These are passed to the `datasets.load_dataset` function.
            wrap: Whether to wrap text sequences.
            train_split: The split to use for the training data. Set to None to skip training data.
            val_split: The split to use for the validation data. Set to None to skip validation data.
            test_split: The split to use for the test data. Set to None to skip test data.
        """
        super().__init__()
        self._dataset_name = dataset_name
        self.hf_dataset_kwargs = hf_dataset_kwargs
        self.local_dataset_kwargs = local_dataset_kwargs
        self.global_batch_size = global_batch_size
        self.prepare_data_per_node = False  # controls whether prepare_data is called on each node or not
        self.collation_strategy = collation_strategy

        self.manual_cache_dir = manual_cache_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.block_size = block_size
        # self.max_length = getattr(tokenizer, "model_max_length", None)
        # if collation_strategy == "pad_truncate" and self.max_length is None:
        #    raise ValueError(
        #        "Tokenizer need to have `model_max_length` attribute to use `pad_truncate` collation strategy."
        #    )
        self.max_length = self.block_size
        self.detokenizer = DETOKENIZERS[detokenizer] if detokenizer else None
        self.tokenizer = tokenizer

        self.tokenizer = tokenizer
        self.text_field_name = text_field_name
        self.num_dataset_workers = num_dataset_workers
        self.streaming = hf_dataset_kwargs.get("streaming", False)
        self.rewrite_manual_cache = rewrite_manual_cache
        if self.streaming:
            raise NotImplementedError(
                "Streaming is not fully supported yet because of the sampler state."
            )
        # get the dataset name from hf_dataset_kwargs
        self._eos_token_id = tokenizer.eos_token_id
        self._bos_token_id = tokenizer.bos_token_id
        self._pad_token_id = tokenizer.pad_token_id
        if self._pad_token_id is None:
            raise ValueError("PAD token is not set.")
        if self.collation_strategy == "wrap":
            if self._bos_token_id is None:
                raise ValueError(
                    "BOS token is not set. This is required for wrapping."
                )
            if self._eos_token_id is None:
                raise ValueError(
                    "EOS token is not set. This is required for wrapping."
                )
        self.columns_to_remove = list(
            set((columns_to_remove or []) + [self.text_field_name])
        )

        self.train_dataset: Optional[
            Union[datasets.Dataset, datasets.IterableDataset]
        ] = None
        self.val_dataset: Optional[
            Union[datasets.Dataset, datasets.IterableDataset]
        ] = None
        self.test_dataset: Optional[
            Union[datasets.Dataset, datasets.IterableDataset]
        ] = None
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs
        self.test_dataloader_kwargs = test_dataloader_kwargs
        # TODO (batch_size): Make sure that global_batch_size = batch_size * num_nodes * num_gpus * accum_steps
        # This needs to be checked outside the datamodule because we don't have access to full training config here.
        # check mdlm get_dataloader for reference

    # region: naming
    @property
    def dataset_name(self):
        """Unique identifier for the dataset used for caching."""
        return self._dataset_name

    def get_cache_file(self, split: str) -> Optional[Path]:
        cache_dir = self.get_cache_dir()
        block_size = self.block_size
        if cache_dir is None:
            return None
        if self.collation_strategy == "wrap":
            return (
                cache_dir
                / f"split={split}__block_size={block_size}__wrapped.dat"
            )
        elif self.collation_strategy == "pad_truncate":
            return (
                cache_dir
                / f"split={split}__max_length={self.max_length}__padded.dat"
            )
        elif self.collation_strategy == "pad":
            # for pad without truncation, we cache the unpadded dataset
            return cache_dir / f"split={split}__unpadded.dat"
        else:
            raise ValueError(
                f"Invalid collation strategy: {self.collation_strategy}"
            )

    def get_cache_dir(self) -> Optional[Path]:
        if (manual_cache_dir := self.manual_cache_dir) is None:
            return None
        else:
            return Path(manual_cache_dir) / self.dataset_name  # type: ignore

    def get_split_name_from_stage(
        self, stage: Optional[str] = None
    ) -> Optional[str]:
        if stage == "fit":
            return self.train_split
        elif stage == "validate":
            return self.val_split
        elif stage == "test":
            return self.test_split

    # endregion: naming
    # region: Preprocessing methods applied using datasets.map

    def tokenize(
        self,
        examples: Mapping[str, Any],
    ) -> Dict[Literal["input_ids", "attention_mask"], List[int]]:
        text = examples[self.text_field_name]
        if self.collation_strategy == "wrap":
            # Based on SEDD and MDLM code.
            # we don't add special tokens through the tokenizer
            # manually add EOS here and let group_texts add BOS to form
            # wrapped batches:
            #  [BOS] sent1 [EOS] sent2-fragment [EOS]
            #  [BOS] sent2-fragment [EOS] sent3 [EOS]
            tokenizer_output = self.tokenizer(
                text,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            tokens = {
                "input_ids": [
                    seq + [self._eos_token_id]
                    for seq in tokenizer_output["input_ids"]  # type: ignore
                ]
            }
        elif (
            self.collation_strategy == "pad"
        ):  # leave padding to collator in the case of variable max_length
            # We let the tokenizer add both EOS and BOS.
            tokens = self.tokenizer(
                text,
                padding=False,
                truncation=False,
                add_special_tokens=True,  # we assume
                return_attention_mask=True,
                return_token_type_ids=False,
            )
        elif (
            self.collation_strategy == "pad_truncate"
        ):  # we can pad till max_length
            # We let the tokenizer add both EOS and BOS.
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=False,
            )
        else:
            raise ValueError(
                f"Invalid collation strategy: {self.collation_strategy}"
            )
        return tokens  # type: ignore

    def group_texts(
        self, examples: Dict[str, List[List[int]]], block_size: int
    ) -> Dict[str, List[List[int]]]:
        eos = self._eos_token_id
        bos = self._bos_token_id
        pad = self._pad_token_id
        real_block_size = block_size - 2  # make space for bos and eos
        # colapse the sequences into a single list of tokens and then create blocks of real_block_size
        input_ids = []
        attention_mask = []
        for block in chunked(flatten(examples["input_ids"]), real_block_size):
            s = [bos] + list(block) + [eos]
            ls = len(s)
            attn = [True] * ls
            s += [pad] * (block_size - ls)
            attn += [False] * (block_size - ls)
            input_ids.append(s)
            attention_mask.append(attn)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def preprocess(self, dataset: DatasetT, desc_prefix: str = "") -> DatasetT:
        streaming = self.streaming
        map_kwargs: Dict[str, Any] = {"batched": True}
        num_processes = (
            self.num_dataset_workers
            if self.num_dataset_workers is not None
            else get_num_processes()
        )

        ops: List[Tuple[str, Callable]] = []

        def _tokenize(
            examples: Mapping[str, Any]
        ) -> Mapping[Literal["input_ids", "attention_mask"], List[int]]:
            text = examples[self.text_field_name]  # assume batched
            if self.detokenizer is not None:
                text = [self.detokenizer(t) for t in text]
            tokenized_text: Dict[
                Literal["input_ids", "attention_mask"], List[int]
            ] = self.tokenize({self.text_field_name: text})
            return tokenized_text

        ops.append(("Tokenizing dataset", _tokenize))
        if self.collation_strategy == "wrap":
            ops.append(("Chunking dataset", self.group_texts))

        if not streaming:
            map_kwargs["num_proc"] = num_processes
            map_kwargs["load_from_cache_file"] = False

            def get_map_kwargs(desc):
                map_kwargs["desc"] = desc_prefix + desc
                return map_kwargs

            for op_name, op in ops:
                dataset = dataset.map(op, **get_map_kwargs(op_name))
        else:
            for op_name, op in ops:
                dataset = dataset.map(op, **map_kwargs)
        # remove columns
        dataset = dataset.remove_columns(self.columns_to_remove)
        return dataset

    # endregion

    # region: Loading dataset
    def load_dataset(self, split: Optional[str] = None) -> DatasetT:
        """Loads the raw dataset in the memory. In typical training pipeline, this method will not be called directly but through get_dataset.

        For non-streaming datasets, this method will typically only be called once, then get_dataset will use the processed and cached dataset.

        Note:
            Might have to override this if the dataset does not support the split argument to select a subset of the data.
        """
        if self.dataset_name in LOCAL_DATASETS:
            return LOCAL_DATASETS[self.dataset_name](
                self.dataset_name, self.local_dataset_kwargs, split
            )
        kwargs = deepcopy(self.hf_dataset_kwargs)
        if split is not None:
            if self.hf_dataset_kwargs.get("split") is not None:
                logger.warning(
                    "split is provided in both hf_dataset_kwargs and load_dataset. Overriding hf_dataset_kwargs split."
                )
            kwargs["split"] = split

        dataset = datasets.load_dataset(**kwargs)
        return cast(DatasetT, dataset)

    def get_dataset(
        self, split: Optional[str] = None
    ) -> Union[datasets.Dataset, datasets.IterableDataset]:
        """
        Returns the cached and preprocessed dataset for the given split for non-streaming datasets.
        This method expects to run on all ranks.
        """
        if not self.streaming and self.manual_cache_dir is not None:
            # The dataset must already be cached by the prepare_data method
            if split is not None:
                cache_file = self.get_cache_file(split=split)
                assert (
                    cache_file is not None
                ), "Call prepare_data from rank 0 before setup."
                logger.info(
                    f"Loading {split} split from cache file {cache_file}"
                )
                dataset = datasets.load_from_disk(str(cache_file))
        else:
            dataset = self.load_dataset(split=split)
            dataset = self.preprocess(dataset, desc_prefix=f"[{split} split] ")
        return dataset  # type: ignore

    # endregion: Loading dataset
    # region: utils
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
                "Are you calling `LightningDataModule.*_dataloader()` methods manually?"
            )
            return False

    def get_collate_fn(
        self, dataset: DatasetT
    ) -> Tuple[Optional[Callable], DatasetT]:
        if self.collation_strategy == "pad":
            return (
                create_collate_fn_for_hf_data_using_pretrained_tokenizer(
                    self.tokenizer
                ),
                dataset,
            )
        else:
            # for the other two strategies, we can use the default collate function
            # default collate requires tensors and not lists so we call with_format("torch")
            return None, dataset.with_format("torch")

    def _ensure_trainer_set(self):
        if self.trainer is None:
            raise ValueError("Trainer is not setup.")

    def _check_grad_accum(self):
        if self.trainer is None:
            raise ValueError("Trainer is not setup. Cannot check grad accum.")
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

    def _determine_per_device_batch_size(self, batch_size: int) -> int:
        if self.trainer is None:
            raise ValueError(
                "Trainer is not setup. Cannot determine the number of devices."
            )
        if self._is_ddp():
            num_nodes = self.trainer.num_nodes
            num_gpus_per_node = self.trainer.num_devices
            accum_steps = self.trainer.accumulate_grad_batches
            per_device_batch_size = batch_size // (
                num_nodes * num_gpus_per_node * accum_steps
            )
            remainder = batch_size % (
                num_nodes * num_gpus_per_node * accum_steps
            )
            if remainder != 0:
                raise ValueError(
                    "Batch size is not divisible by the number of nodes, GPUs per node and accum_steps."
                )
            return per_device_batch_size
        else:
            return batch_size

    # endregion: utils

    # region: Mandatory Lightning DataModule methods

    def prepare_data(self):
        """Do global, one-time processing here.
        Note (general):
            This method is called before setup() is called on each node. There is a barrier after this method if it is called on all nodes.
            If self.prepare_data_per_node is True, this method is called on each node otherwise it is called once on the main node.
            Since this only runs on the main process, do not assign any state to self here.

        We use this method to download the dataset and process it for the training only when manual caching is enabled.
        Note, there cannot be any caching when streaming.
        This includes the following:
        1. Download all three splits
        2. Tokenizer
        3. Wrap text sequences if needed
        """
        # check if we have a manual cache dir for this dataset
        if self.streaming:
            ranked_logger.info(
                f"Skipping data preparation for {self.dataset_name} because streaming is enabled."
            )
            return
        if self.manual_cache_dir is None:
            # just call load_dataset and exit
            ranked_logger.info(
                f"Downloading the dataset for {self.dataset_name}"
            )
            for split_stage, split in zip(
                ("train", "val", "test"),
                (self.train_split, self.val_split, self.test_split),
            ):
                if split is not None:
                    self.load_dataset(split=split)
            return
        ranked_logger.info(f"Preprocessing data for {self.dataset_name}")
        dataset_dict = {}

        for split_stage, split in zip(
            ("train", "val", "test"),
            (self.train_split, self.val_split, self.test_split),
        ):
            if split is not None:
                cache_file = self.get_cache_file(split=split)
                # manual cache enabled
                if cache_file is not None:
                    # manual cache enabled and cache file exists
                    if cache_file.exists():
                        if not self.rewrite_manual_cache:
                            # nothing to preprocess
                            ranked_logger.info(
                                f"Cache file {cache_file} for {split} already exists. Nothing to preprocess."
                            )
                            continue
                        else:
                            ranked_logger.info(
                                f"Cache file {cache_file} for {split} already exists. "
                                f"Rewriting because rewrite_manual_cache is set to True."
                            )
                    else:
                        # manual cache enabled and but cache file does not exist
                        ranked_logger.info(
                            f"Cache file {cache_file} for {split} split does not exist. Processing the split."
                        )

                _dataset_split = self.load_dataset(split=split)
                dataset_split = self.preprocess(
                    _dataset_split, desc_prefix=f"[{split} split] "
                )
                # now cache the split
                if cache_file is not None:
                    ranked_logger.info(
                        f"Caching {split} split to {cache_file}"
                    )
                    dataset_split.save_to_disk(str(cache_file))
                else:  # no manual cache dir provided
                    ranked_logger.warning(
                        "No manual cache dir provided. Skipping caching."
                    )
            else:
                ranked_logger.warning(
                    f"No split provided for {split_stage}. Skipping."
                )

        return dataset_dict

    def setup(self, stage: Optional[str] = None):
        # TODO: shuffle when streaming 'train'? See https://huggingface.co/docs/datasets/v3.1.0/en/stream#shuffle
        # TODO: Create individual dataloaders here
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                self.train_dataset = self.get_dataset(self.train_split)
            if self.val_split is not None and self.val_dataset is None:
                self.val_dataset = self.get_dataset(self.val_split)

        if stage == "validate":
            if self.val_split is not None and self.val_dataset is None:
                self.val_dataset = self.get_dataset(self.val_split)

        if stage == "test":
            if self.test_split is not None and self.test_dataset is None:
                self.test_dataset = self.get_dataset(self.test_split)
            else:
                raise NotImplementedError("No test split provided.")

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset not loaded. Call setup() first.")
        # collate function
        collate_fn, self.train_dataset = self.get_collate_fn(self.train_dataset)  # type: ignore
        # We setup the generator here because we want its seed to be
        # be dependent on the global seed. Letting the sampler create
        # its own generator will use a hardcoded seed of 0 as of today (https://github.com/pytorch/data/issues/1440)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        # TODO (fault-tolerance): These samplers use naive fast forwarding, which might be slow.
        if self._is_ddp():

            # per_device_batch_size = self._determine_per_device_batch_size(
            #    self.train_dataloader_kwargs["batch_size"]
            # )
            # CLEANUP: Don't need to determine per-device batch size here.
            self._check_grad_accum()
            per_device_batch_size = self.train_dataloader_kwargs["batch_size"]
            train_dataloader_kwargs = deepcopy(self.train_dataloader_kwargs)
            train_dataloader_kwargs["batch_size"] = per_device_batch_size
            logger.info(
                f"DDP: Using per-device batch size of {per_device_batch_size} for training"
            )
            logger.info(
                "Detected DDP strategy. Using StatefulDistributedSampler with StatefulDataLoader for training"
            )
            shuffle = train_dataloader_kwargs.pop(
                "shuffle", False
            )  # can't have shuffle passed when using Sampler
            sampler = StatefulDistributedSampler(
                self.train_dataset, seed=seed, shuffle=shuffle
            )
            dataloader = StatefulDataLoader(
                self.train_dataset,
                sampler=sampler,
                collate_fn=collate_fn,
                **train_dataloader_kwargs,
            )
            self.train_dataloader_kwargs = train_dataloader_kwargs
        else:
            logger.info(
                "DDP strategy was not detected. Using StatefulRandomSampler with StatefulDataLoader for training"
            )
            generator = torch.Generator().manual_seed(seed)
            # StatefulRandomSampler is the default internal sampler for StatefulDataLoader
            dataloader = StatefulDataLoader(
                self.train_dataset,  # type: ignore
                generator=generator,
                collate_fn=collate_fn,
                **self.train_dataloader_kwargs,
            )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset not loaded. Call setup() first."
            )
        self._ensure_trainer_set()
        collate_fn, self.val_dataset = self.get_collate_fn(self.val_dataset)  # type: ignore
        # make sure eval batch size is divisible by num_gpus_per_node so that all GPUs end up with the same number of samples
        if self._is_ddp():
            if (
                self.val_dataloader_kwargs["batch_size"]
                % self.trainer.num_devices  # type: ignore[attr-defined]
                != 0
            ):
                raise ValueError(
                    "Validation batch size is not divisible by the number of GPUs per node."
                )
            # CLEANUP: Don't need to determine per-device batch size here.
            # per_device_batch_size = self._determine_per_device_batch_size(
            #    self.val_dataloader_kwargs["batch_size"]
            # )
            # val_dataloader_kwargs = deepcopy(self.val_dataloader_kwargs)
            # val_dataloader_kwargs["batch_size"] = per_device_batch_size
            # logger.info(
            #    f"DDP: Using per-device batch size of {per_device_batch_size} for validation"
            # )
            val_dataloader_kwargs = deepcopy(self.val_dataloader_kwargs)
        else:
            val_dataloader_kwargs = deepcopy(self.val_dataloader_kwargs)
        val_dataloader = DataLoader(
            self.val_dataset,  # type: ignore
            collate_fn=collate_fn,
            **val_dataloader_kwargs,
        )
        self.val_dataloader_kwargs = val_dataloader_kwargs
        return val_dataloader

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataloader_kwargs is None:
            return None
        if self.test_dataset is None:
            raise ValueError("Test dataset not loaded. Call setup() first.")
        collate_fn, self.test_dataset = self.get_collate_fn(self.test_dataset)  # type: ignore
        # make sure eval batch size is divisible by num_gpus_per_node so that all GPUs end up with the same number of samples
        if (
            self._is_ddp()
            and self.test_dataloader_kwargs["batch_size"]
            % self.trainer.num_devices  # type: ignore[attr-defined]
            != 0
        ):
            raise ValueError(
                "Test batch size is not divisible by the number of GPUs per node."
            )
        return DataLoader(
            self.test_dataset,  # type: ignore
            collate_fn=collate_fn,
            **self.test_dataloader_kwargs,
        )

    # endregion


# endregion: DataModule
