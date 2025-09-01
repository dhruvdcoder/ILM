import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import datasets
import torch
from jaxtyping import Integer
from tokenizers import processors
from torch import Tensor as TT
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AddedToken, PreTrainedTokenizer

from pcdd import flags
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only

from .datamodule import (
    ARLMBatch,
    ARLMTokenizerMixin,
    BaseCollatorInput,
    BaseDataModule,
    Collator,
    DataLoaderKwargs,
    DefaultIDLMCollator,
    DefaultIDLMCollatorForPrediction,
    DefaultILMCollator,
    DefaultILMCollatorForPrediction,
    DefaultILMWithLengthClassificationCollator,
    DefaultILMWithLengthClassificationCollatorForPrediction,
    DefaultMDLMCollator,
    DefaultMDLMCollatorForPrediction,
    DefaultARLMCollator,
    DefaultARLMCollatorForPrediction,
    IDLMBatch,
    IDLMTokenizerMixin,
    ILMBatch,
    MDLMBatch,
    MDLMTokenizerMixin,
    Tokenizer,
    print_batch_arlm,
    print_batch_idlm,
    print_batch_ilm,
    print_batch_mdlm,
    print_batch_arlm,
    pad_prefix_suffix,
)
from typing import cast

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


def batched(iterable: List, n=3, pad=-1):
    l_iterable = len(iterable)
    # pad the iterable to make it divisible by n
    num_pads = (n - l_iterable % n) % n
    if num_pads != 0:
        iterable = iterable + [pad] * num_pads
        l_iterable = len(iterable)
    assert len(iterable) % n == 0
    for ndx in range(0, l_iterable, n):
        yield iterable[ndx : min(ndx + n, l_iterable)]


########################################################
# region: Types


# endregion: Types
########################################################


########################################################
# region: Tokenization


class ILMTokenizerMixin(IDLMTokenizerMixin):
    """Just updates the build_inputs_with_special_tokens and post_processor  to add BOS after prefix."""

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # [CLS] [prefix] [BOS] [non_prefix]
        # 0     1........ 1    2........2
        # token_ids_1 is assumed to be a prefix
        if token_ids_1 is not None:
            return (
                [self.cls_token_id]  # type: ignore
                + token_ids_1
                + [self.bos_token_id]  # type: ignore
                + token_ids_0
            )  # type: ignore
        else:
            return [self.cls_token_id] + [self.bos_token_id] + token_ids_0  # type: ignore

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

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        if skip_special_tokens:
            # only skip mask and pad tokens for readability
            to_skip = {
                self.mask_token,
                self.pad_token,
            }
            tokens = [t for t in tokens if t not in to_skip]
        return "".join(tokens)


class MDLMTokenizerMixinZebra(MDLMTokenizerMixin):
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is not None:
            return (
                token_ids_1 + [self.bos_token_id] + token_ids_0  # type: ignore
            )  # type: ignore
        else:
            return [self.bos_token_id] + token_ids_0  # type: ignore

    def create_post_processor(self) -> processors.TemplateProcessing:
        if (self.bos_token is None) or (self.bos_token_id is None):
            raise ValueError("bos_token is required.")
        post_processor = processors.TemplateProcessing(
            single=f"{self.bos_token}:1 $A:2",
            pair=f"$B:1 {self.bos_token}:1 $A:2",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
            ],
        )
        return post_processor

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        if skip_special_tokens:
            # only skip mask and pad tokens for readability
            to_skip = {
                self.mask_token,
                self.pad_token,
            }
            tokens = [t for t in tokens if t not in to_skip]
        return "".join(tokens)


# fmt: off
TOKENIZER_CONFIG = {
    "vocab": [
        "0", "1", "2", "3", "4", "5", # digits
        "nbr", "left-of", "inbetween", "immediate-left", "ends", # operators
        "!=", "=",  "c", "n", 
        "CLUE_END", "RHS", "LHS",
        ], "model_max_length": 460}
# fmt: on


class ZebraTokenizer(PreTrainedTokenizer, Tokenizer):
    model_input_names: List[str] = ["input_ids", "attention_mask"]

    def __init__(self, vocab: Sequence[str], model_max_length: int, **kwargs):
        """
        Args:
            vocab (Sequence[str]): List of desired tokens. Following are list of all of the special tokens with
                their corresponding ids:
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[MASK]": 2,
                    "[EOS]": 3,
                    "[BOS]": 4,
                an id (starting at 5) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        self.vocab = vocab
        self.model_max_length = model_max_length
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        self._vocab_str_to_int = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[MASK]": 2,
            "[EOS]": 3,
            "[BOS]": 4,
            "[SEP]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(vocab)},
        }
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()
        }

        super().__init__(
            eos_token=eos_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            bos_token=bos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        # suppose text is a like "split1 split2 split3", convert to character if split* not in vocab
        return text.split(" ")

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int[token]  # let it raise keyerror

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def get_config(self) -> Dict:
        return {
            "vocab": self.vocab,
            "model_max_length": self.model_max_length,
        }

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    @classmethod
    def from_config(cls, config: Dict) -> "ZebraTokenizer":
        cfg = {}
        cfg["vocab"] = config["vocab"]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(
        self, save_directory: Union[str, os.PathLike], **kwargs
    ):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(
        cls, save_directory: Union[str, os.PathLike], **kwargs
    ):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

    @classmethod
    def from_default_config(cls):
        return cls.from_config(TOKENIZER_CONFIG)

    def get_solution_set(
        self, token_ids: List[int]
    ) -> Set[Tuple[int, int, int]]:
        """
        Get the solution set for a given ids.
        """
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        # remove special tokens
        to_skip = {
            self.mask_token_id,
            self.pad_token_id,
            self.cls_token_id,
        }
        token_ids = [t for t in _token_ids if t not in to_skip]
        # Find the BOS token
        bos_found = False
        for i, idx in enumerate(token_ids):
            if idx == self.bos_token_id:
                bos_found = True
                break
        if not bos_found:
            raise ValueError("BOS token not found")
        # Go over the tokens after BOS and create Tuples of triples
        solution_set = set()
        for entity_id, attr_id, value_id in batched(token_ids[i + 1 :], 3):
            solution_set.add((entity_id, attr_id, value_id))
        return solution_set

    def get_solution_sets_from_batch(
        self, token_ids: Integer[TT, " *batch seq_len"]
    ) -> List[Set[Tuple[int, int, int]]]:
        """
        Get the solution sets for a batch of ids.
        """
        _token_ids: List[List[int]] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        return [self.get_solution_set(token_ids) for token_ids in _token_ids]


class ILMZebraTokenizer(ILMTokenizerMixin, ZebraTokenizer):
    pass


class MDLMZebraTokenizer(MDLMTokenizerMixinZebra, ZebraTokenizer):
    pass


class ARLMZebraTokenizer(ARLMTokenizerMixin, ZebraTokenizer):
    pass


# endregion: Tokenization
########################################################


########################################################
# region: DataModule


class ZebraILMCollator(DefaultILMCollator):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # shift the end of prefix to left by 1, by roll and fill
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0

        return batch


class ZebraIDLMCollator(DefaultIDLMCollator):
    """Add constraint to the batch"""

    def __call__(self, examples: List[BaseCollatorInput]) -> IDLMBatch:
        batch: IDLMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # shift the end of prefix to left by 1, by roll and fill
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class ZebraIDLMCollatorForPrediction(DefaultIDLMCollatorForPrediction):
    def __call__(self, examples: List[BaseCollatorInput]) -> IDLMBatch:
        batch: IDLMBatch = super().__call__(examples)
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class ZebraILMWithLengthClassificationCollator(
    DefaultILMWithLengthClassificationCollator
):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class ZebraILMCollatorForPrediction(DefaultILMCollatorForPrediction):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class ZebraILMWithLengthClassificationCollatorForPrediction(
    DefaultILMWithLengthClassificationCollatorForPrediction
):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class ZebraMDLMCollator(DefaultMDLMCollator):
    def __call__(self, examples: List[BaseCollatorInput]) -> MDLMBatch:
        batch: MDLMBatch = super().__call__(examples)
        return batch


class ZebraMDLMCollatorForPrediction(DefaultMDLMCollatorForPrediction):
    def __call__(self, examples: List[BaseCollatorInput]) -> MDLMBatch:
        batch: MDLMBatch = super().__call__(examples)
        return batch


class ZebraARLMCollator(DefaultARLMCollator):
    def __call__(self, examples: List[BaseCollatorInput]) -> ARLMBatch:
        batch: ARLMBatch = super().__call__(examples)
        return batch


class ZebraARLMCollatorForPrediction(DefaultARLMCollatorForPrediction):
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
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        base_batch = pad_prefix_suffix(self.tokenizer, examples, max_seq_len)
        base_batch["constraint"] = (
            (base_batch["token_type_ids"] == 1)
            | (base_batch["input_ids"] == self.tokenizer.pad_token_id)
        ).long()
        base_batch["drop"] = (1 - base_batch["constraint"]).long()
        base_batch = cast(ARLMBatch, base_batch)
        return base_batch


def _get_default_dataloader_kwargs(
    type: Literal["train", "val", "test", "predict"],
) -> DataLoaderKwargs:
    return {
        "batch_size": 128,
        "num_workers": 4,
        "shuffle": True if type == "train" else False,
        "pin_memory": True,
    }


def preprocess_fn(
    examples: Dict[Literal["input", "output"], List[List[str]]],
    tokenizer,
    max_length=460,
) -> Dict[
    Literal["input_ids", "attention_mask", "token_type_ids"], List[List[int]]
]:

    tokenized_examples = tokenizer(
        examples["output"],
        examples["input"],
        return_token_type_ids=True,
    )
    return tokenized_examples


def preprocess_fn_sorted(
    examples, tokenizer, max_length=460
) -> Dict[
    Literal["input_ids", "attention_mask", "token_type_ids"], List[List[int]]
]:
    tokenized_examples = tokenizer(
        examples["sorted_output"],
        examples["input"],
        return_token_type_ids=True,
    )
    return tokenized_examples


class ZebraDataModule(BaseDataModule):
    prepare_data_per_node: bool = False

    def get_default_collator(
        self,
        tokenizer: ZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("Not implemented")

    def get_default_prediction_collator(
        self,
        tokenizer: ZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("Not implemented")

    def __init__(
        self,
        manual_cache_dir: str,
        tokenizer: ZebraTokenizer,
        noise_schedule: NoiseSchedule,
        train_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        val_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        test_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        predict_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        rewrite_manual_cache: bool = False,
        global_batch_size: Optional[int] = None,
        block_size: int = 80,
        num_dataset_workers: int = 4,
        data_format: Literal["base", "sorted"] = "base",
    ):
        super().__init__()
        rewrite_manual_cache = True
        self.data_format = data_format
        if data_format == "base":
            self.dataset_name = "dhruveshpatel/zebra-puzzles"
        elif data_format == "sorted":
            self.dataset_name = "AvinashAmballa/zebra-puzzles-sorted"
        else:
            raise ValueError(f"Invalid data format: {data_format}")

        self.manual_cache_dir = manual_cache_dir
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.rewrite_manual_cache = rewrite_manual_cache
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.train_dataloader_kwargs: DataLoaderKwargs = (
            train_dataloader_kwargs or _get_default_dataloader_kwargs("train")
        )
        self.val_dataloader_kwargs: DataLoaderKwargs = (
            val_dataloader_kwargs or _get_default_dataloader_kwargs("val")
        )
        self.test_dataloader_kwargs: DataLoaderKwargs = (
            test_dataloader_kwargs or _get_default_dataloader_kwargs("test")
        )
        self.predict_dataloader_kwargs: DataLoaderKwargs = (
            predict_dataloader_kwargs
            or _get_default_dataloader_kwargs("predict")
        )
        self.collator = self.get_default_collator(
            tokenizer, block_size, noise_schedule
        )
        self.prediction_collator = self.get_default_prediction_collator(
            tokenizer, block_size, noise_schedule
        )
        self.train_dataloader_names = {
            0: "lm",
            1: "prediction",
        }
        self.val_dataloader_names = {
            0: "lm",
            1: "prediction",
        }
        self.test_dataloader_names = {
            0: "lm",
            1: "prediction",
        }
        self.predict_dataloader_names = {
            1: "prediction",
        }
        self.num_dataset_workers = num_dataset_workers
        self.block_size = block_size
        if global_batch_size is not None:
            logger.warning(
                "Global batch size will be ignore. We don't support DDP for Countdown."
            )

    def set_epoch(self, epoch: int):
        pass  # nothing to do here

    def prepare_data(self):
        # check for manual cache
        cache_dir = Path(self.manual_cache_dir) / self.dataset_name
        if not isinstance(self.tokenizer, ILMZebraTokenizer):
            cache_dir = (
                Path(self.manual_cache_dir)
                / self.dataset_name
                / self.tokenizer.__class__.__name__
            )
        _cached = False
        if cache_dir.exists():
            _cached = True
        if _cached and not self.rewrite_manual_cache:
            logger.info(f"Loading dataset from manual cache at {cache_dir}")
            return

        _datasets = datasets.load_dataset(self.dataset_name)
        _datasets = _datasets.map(
            (
                preprocess_fn
                if self.data_format == "base"
                else preprocess_fn_sorted
            ),
            batched=True,
            num_proc=self.num_dataset_workers,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "max_length": self.block_size,
            },
            remove_columns=[
                "input",
                ("output" if self.data_format == "base" else "sorted_output"),
            ],
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset to manual cache at {cache_dir}")
        _datasets.save_to_disk(cache_dir)

    def setup(self, stage: Optional[str] = None):
        cache_dir = Path(self.manual_cache_dir) / self.dataset_name
        if not isinstance(self.tokenizer, ILMZebraTokenizer):
            cache_dir = (
                Path(self.manual_cache_dir)
                / self.dataset_name
                / self.tokenizer.__class__.__name__
            )
        logger.info(f"Loading dataset from manual cache at {cache_dir}")
        _datasets = datasets.load_from_disk(cache_dir)
        self.train_dataset = _datasets["train"]
        self.val_dataset = _datasets["test"]
        self.val_prediction_dataset = _datasets["test"].select(
            range(5 * self.predict_dataloader_kwargs["batch_size"])
        )
        self.test_dataset = _datasets["test"]

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not loaded. Call setup() first.")
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator().manual_seed(seed)
        assert self.train_dataset is not None
        dataloader = StatefulDataLoader(
            self.train_dataset,
            generator=generator,
            collate_fn=self.collator,
            **self.train_dataloader_kwargs,
        )
        return dataloader

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Val dataset not loaded. Call setup() first.")
        if not flags.DEBUG_OVERFIT_TRAIN:
            lm_dataloader = DataLoader(
                self.val_dataset,
                collate_fn=self.collator,
                **self.val_dataloader_kwargs,
            )
            prediction_dataloader = DataLoader(
                self.val_prediction_dataset,
                collate_fn=self.prediction_collator,
                **self.val_dataloader_kwargs,
            )
        else:
            lm_dataloader = DataLoader(
                self.train_dataset,
                collate_fn=self.collator,
                **self.val_dataloader_kwargs,
            )
            prediction_dataloader = DataLoader(
                self.train_dataset,
                collate_fn=self.prediction_collator,
                **self.val_dataloader_kwargs,
            )
        return [lm_dataloader, prediction_dataloader]

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not loaded. Call setup() first.")
        lm_dataloader = DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            **self.test_dataloader_kwargs,
        )
        prediction_dataloader = DataLoader(
            self.test_dataset,
            collate_fn=self.prediction_collator,
            **self.test_dataloader_kwargs,
        )
        return [lm_dataloader, prediction_dataloader]


class ZebraDataModuleForILM(ZebraDataModule):
    def get_default_collator(
        self,
        tokenizer: ZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ZebraILMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: ZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ZebraILMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: ILMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_ilm(self, batch, split, dataloader_idx)


class ZebraDataModuleForILMWithLengthClassification(ZebraDataModuleForILM):
    def get_default_collator(
        self,
        tokenizer: ZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ZebraILMWithLengthClassificationCollator(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_collator(
        self,
        tokenizer: ZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ZebraILMWithLengthClassificationCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )


class ZebraDataModuleForIDLM(ZebraDataModule):
    def get_default_collator(
        self,
        tokenizer: ZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ZebraIDLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: ZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ZebraIDLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: IDLMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_idlm(self, batch, split, dataloader_idx)


class ZebraDataModuleForARLM(ZebraDataModule):
    def get_default_collator(
        self,
        tokenizer: ARLMZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ZebraARLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: ARLMZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ZebraARLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: ARLMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_arlm(self, batch, split, dataloader_idx)


class ZebraDataModuleForMDLM(ZebraDataModule):
    def get_default_collator(
        self,
        tokenizer: MDLMZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMDLMCollator(
            tokenizer, block_size, noise_schedule, loss_on_padding=True
        )

    def get_default_prediction_collator(
        self,
        tokenizer: MDLMZebraTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMDLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: MDLMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_mdlm(self, batch, split, dataloader_idx)


# endregion: DataModule
########################################################
