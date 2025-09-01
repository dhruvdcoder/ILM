import json
import os
import random
from pathlib import Path
import re
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
)
from torch import Tensor as TT
from jaxtyping import Integer
import datasets
import numpy as np
import torch
from tokenizers import processors
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AddedToken, PreTrainedTokenizer

from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only

from .datamodule import (
    BaseCollatorInput,
    BaseDataModule,
    Collator,
    DataLoaderKwargs,
    DefaultILMCollator,
    DefaultILMCollatorForPrediction,
    DefaultILMWithLengthClassificationCollator,
    DefaultILMWithLengthClassificationCollatorForPrediction,
    IDLMTokenizerMixin,
    ILMBatch,
    Tokenizer,
    print_batch_ilm,
)

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


########################################################
# region: Types


# endregion: Types
########################################################


########################################################
# region: Tokenization

# fmt: off
tokenizer_config = {
    "vocab": [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", # digits
        "*", "+", "-", "/", "=", # operators
        ",", # comma
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" # letters
        ], "model_max_length": 80}
# fmt: on


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

    def batch_convert_ids_to_expressions(
        self, token_ids: Integer[TT, "batch_size seq_len"]
    ) -> Dict[str, Any]:
        token_ids_: List[List[int]] = token_ids.tolist()  # type: ignore
        dict = {
            "input_numbers": [],
            "output_numbers": [],
            "output_expressions": [],
        }
        for _token_ids in token_ids_:
            res = self.convert_ids_to_expressions(_token_ids)
            for key in dict:
                dict[key].append(res[key])
        return dict

    def convert_ids_to_expressions(self, token_ids: List[int]) -> Dict:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        # only skip mask and pad tokens for readability
        to_skip = {
            self.mask_token,
            self.pad_token,
            self.cls_token,
        }
        tokens = [t for t in tokens if t not in to_skip]
        input_tokens = []
        output_tokens = []
        bos_token_found = False
        for token in tokens:
            if token == self.bos_token:
                bos_token_found = True
                continue
            if bos_token_found:
                output_tokens.append(token)
            else:
                input_tokens.append(token)
        if not bos_token_found:
            raise ValueError(f"Invalid expression: {tokens}")
        input_text = "".join(input_tokens)
        output_text = "".join(output_tokens)
        input_numbers = [int(n) for n in input_text.split(",")]
        output_numbers = list(map(int, re.findall(r"\d+", output_text)))
        expressions = output_text.split(",")
        output_expressions = []
        try:
            for expr in expressions:
                if "=" not in expr:
                    raise ValueError(f"Invalid expression: {expr}")
                lhs_expr, rhs_expr = expr.split("=")
                output_expressions.append([lhs_expr, rhs_expr])
                # try:
                #    lhs = eval(lhs_expr)
                #    rhs = eval(rhs_expr)
                # except Exception as e:
                #    raise ValueError(f"Invalid expression: {expr}") from e
        except Exception as e:
            output_expressions = [
                ["0", "1"]
            ]  # the string is not even well formatted

        return {
            "input_numbers": input_numbers,
            "output_numbers": output_numbers,
            "output_expressions": output_expressions,
        }


class CountdownTokenizer(ILMTokenizerMixin, PreTrainedTokenizer, Tokenizer):
    model_input_names: List[str] = ["input_ids", "attention_mask"]

    def __init__(
        self, vocab: Sequence[str], model_max_length: int = 80, **kwargs
    ):
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
        splits = text.split(" ")
        tokens = []
        for split in splits:
            if split != "":
                if split in self._vocab_str_to_int:
                    tokens.extend([split, " "])
                else:
                    tokens.extend(list(split) + [" "])
        return tokens[:-1]

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int["[UNK]"]
        )

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_config(self) -> Dict:
        return {
            "vocab": self.vocab,
            "model_max_length": self.model_max_length,
        }

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    @classmethod
    def from_config(cls, config: Dict) -> "CountdownTokenizer":
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
        return cls.from_config(tokenizer_config)


# endregion: Tokenization
########################################################


########################################################
# region: DataModule


class CountdownILMCollator(DefaultILMCollator):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # shift the end of prefix to left by 1, by roll and fill
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0

        return batch


class CountdownILMWithLengthClassificationCollator(
    DefaultILMWithLengthClassificationCollator
):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class CountdownILMCollatorForPrediction(DefaultILMCollatorForPrediction):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class CountdownILMWithLengthClassificationCollatorForPrediction(
    DefaultILMWithLengthClassificationCollatorForPrediction
):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


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
    max_length=80,
) -> Dict[
    Literal["input_ids", "attention_mask", "token_type_ids"], List[List[int]]
]:

    tokenized_examples = tokenizer(
        examples["output"],
        examples["input"],
        return_token_type_ids=True,
    )
    return tokenized_examples


class CountdownDataModule(BaseDataModule):
    prepare_data_per_node: bool = False

    def get_default_collator(
        self,
        tokenizer: CountdownTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("Not implemented")

    def get_default_prediction_collator(
        self,
        tokenizer: CountdownTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("Not implemented")

    def __init__(
        self,
        version: Literal["3", "4", "5"],
        manual_cache_dir: str,
        tokenizer: CountdownTokenizer,
        noise_schedule: NoiseSchedule,
        train_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        val_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        test_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        predict_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        rewrite_manual_cache: bool = False,
        global_batch_size: Optional[int] = None,
        block_size: int = 80,
        num_dataset_workers: int = 4,
        # data_format: Literal["standard"] = "standard",
    ):
        super().__init__()
        self.dataset_name = f"dhruveshpatel/countdown-{version}"
        self.num_expressions = int(version) - 1
        self.version = version
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
        datasets.load_dataset(self.dataset_name)

    def setup(self, stage: Optional[str] = None):
        _datasets = datasets.load_dataset(self.dataset_name)
        _datasets = _datasets.map(
            preprocess_fn,
            batched=True,
            num_proc=self.num_dataset_workers,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "max_length": self.block_size,
            },
            remove_columns=["input", "output"],
        )
        self.train_dataset = _datasets["train"]
        self.val_dataset = _datasets["test"]
        self.val_prediction_dataset = _datasets["test"].select(
            range(1 * self.predict_dataloader_kwargs["batch_size"])
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


class CountdownDataModuleForILM(CountdownDataModule):
    def get_default_collator(
        self,
        tokenizer: CountdownTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return CountdownILMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: CountdownTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return CountdownILMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: ILMBatch,
        split: Literal["train", "val", "test", "predict"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_ilm(self, batch, split, dataloader_idx)


class CountdownDataModuleForILMWithLengthClassification(
    CountdownDataModuleForILM
):
    def get_default_collator(
        self,
        tokenizer: CountdownTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return CountdownILMWithLengthClassificationCollator(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_collator(
        self,
        tokenizer: CountdownTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return CountdownILMWithLengthClassificationCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )


# endregion: DataModule
########################################################
