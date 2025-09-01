"""Shared datamodule for all pre-training tasks across all model types."""

from copy import deepcopy
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union, cast

import datasets
import torch
from datasets import Dataset, DatasetDict, IterableDataset
from datasets.distributed import split_dataset_by_node
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase
from tokenizers import Tokenizer as TokenizerHuggingFace
from tokenizers import processors
from tokenizers.models import Unigram
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import UnigramTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.os import get_num_processes
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only

from .datamodule import (
    BaseDataModule,
    Collator,
    DataLoaderKwargs,
    DefaultCollator,
    DefaultCollatorWithDynamicPadding,
    DefaultEmptyDataset,
    DefaultIDLMCollatorForPrediction,
    DefaultIDLMCollatorForPredictionWithDynamicPadding,
    DefaultIDLMCollatorWithDynamicPadding,
    IDLMBatch,
    IDLMTokenizerMixin,
    MLMPredictionCollatorWithPadding,
    PadTruncateProcessor,
    Processor,
    Tokenizer,
    print_batch_base,
    print_batch_idlm,
)

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)

_HF_DATASET_NAME = "vesteinn/babylm"


class IDLMTokenizerForBabylmFast(IDLMTokenizerMixin, PreTrainedTokenizerFast):
    def __init__(
        self,
        *args,
        eos_token: str = "<|eos|>",
        bos_token: str = "<|bos|>",
        cls_token: str = "<|cls|>",
        pad_token: str = "<|pad|>",
        mask_token: str = "<|mask|>",
        sep_token: str = "<|sep|>",
        unk_token: str = "<|unk|>",
        **kwargs,
    ):
        super().__init__(
            *args,
            eos_token=eos_token,
            bos_token=bos_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sep_token=sep_token,
            unk_token=unk_token,
            **kwargs,
        )


def train_babylm_tokenizer(
    dataset: Union[Dataset, DatasetDict],
    vocab_size: int = 16000,
) -> TokenizerHuggingFace:
    def batch_iterator(data, batch_size=1000):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]["text"]

    tokenizer = TokenizerHuggingFace(
        model=Unigram(),  # type: ignore
    )
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<|pad|>",
            "<|mask|>",
            "<|eos|>",
            "<|bos|>",
            "<|cls|>",
            "<|sep|>",
            "<|unk|>",
        ],
        unk_token="<|unk|>",
    )
    tokenizer.train_from_iterator(
        batch_iterator(dataset),
        trainer=trainer,
    )
    return tokenizer


class IdsToExampleProcessor(Processor):
    """
    Converts raw token_ids to input_ids, attention_mask, and token_type_ids.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            example["token_ids"]
        )
        attention_mask = [1] * len(input_ids)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
            example["token_ids"]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


class BabylmDataModule(BaseDataModule):
    manual_cache_dir: Path
    train_dataset_lm: Optional[IterableDataset]
    """The train dataset for the language model.[in setup()]"""
    val_dataset_lm: Optional[Union[IterableDataset, Dataset]]
    """The validation dataset for the language model.[in setup()]"""
    test_dataset_lm: Optional[Union[IterableDataset, Dataset]]
    """The test dataset for the language model.[in setup()]"""
    predict_dataset_lm: Optional[Union[IterableDataset, Dataset]]
    """The predict dataset for the language model.[in setup()]"""

    def __init__(
        self,
        manual_cache_dir: str,
        tokenizer: PreTrainedTokenizerFast,
        noise_schedule: NoiseSchedule,
        train_dataloader_kwargs: DataLoaderKwargs,
        val_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        test_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        predict_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        rewrite_manual_cache: bool = False,
        block_size: int = 128,
        global_batch_size: int = 512,
        num_dataset_workers: Optional[int] = None,
        collator: Optional[Collator] = None,
        prediction_collator: Optional[Collator] = None,
    ):
        self.dataset_name = _HF_DATASET_NAME
        self.train_split = "train"
        self.val_split = "validation"
        self.test_split = "test"
        self.num_shards = 256
        """num_shards for save_to_disk. Keep high for multiprocess dataloading."""
        self.noise_schedule = noise_schedule
        self.block_size = block_size
        self.manual_cache_dir: Path = (
            Path(manual_cache_dir) / self.dataset_name
        )
        self.train_dataset_lm = None  # set in setup()
        self.val_dataset_lm = None  # set in setup()
        self.test_dataset_lm = None  # set in setup()

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
            0: "prediction",
        }
        self.num_preprocessing_workers = (
            num_dataset_workers or get_num_processes()
        )

        self.tokenizer = tokenizer
        self.collator = collator or self.get_default_collator(
            tokenizer, block_size, noise_schedule
        )
        self.collator.tokenizer = tokenizer
        self.prediction_collator = (
            prediction_collator
            or self.get_default_prediction_collator(
                tokenizer, block_size, noise_schedule
            )
        )
        self.prediction_collator.tokenizer = tokenizer
        self.processor = IdsToExampleProcessor(tokenizer)

        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs or {
            **deepcopy(train_dataloader_kwargs),
            "shuffle": False,
        }
        self.test_dataloader_kwargs = test_dataloader_kwargs or {
            **deepcopy(train_dataloader_kwargs),
            "shuffle": False,
        }
        self.predict_dataloader_kwargs = predict_dataloader_kwargs or {
            **deepcopy(train_dataloader_kwargs),
            "shuffle": False,
        }
        self.global_batch_size = global_batch_size
        self.rewrite_manual_cache = rewrite_manual_cache

    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultCollatorWithDynamicPadding(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return MLMPredictionCollatorWithPadding(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_dataset(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> torch.utils.data.IterableDataset:
        return DefaultEmptyDataset(
            tokenizer,
            100,
            empty_text=" ".join([tokenizer.mask_token] * block_size),
        )

    def get_cache_dir(self, split: str) -> Path:
        return self.manual_cache_dir / "raw" / f"{split}"

    def prepare_data(self) -> None:
        """Do global, one-time processing here.
        Note (general):
            This method is called before setup() is called on each node. There is a barrier after this method if it is called on all nodes.
            If self.prepare_data_per_node is True, this method is called on each node otherwise it is called once on the main node.
            Since this only runs on the main process, do not assign any state to self here.

        We will do the following:
        1. Download all three splits from huggingface datasets.
        """
        # check if we have already cached the dataset.
        _cached = True
        for split in ["train", "validation", "test"]:
            _dir = self.get_cache_dir(split)
            if _dir.exists():
                logger.info(f"Found cached {split} dataset at {_dir}")
                if not self.rewrite_manual_cache:
                    logger.info("Skipping preprocessing of cached dataset. ")
                    continue
                else:
                    logger.info("Rewriting cached dataset. ")
                    _cached = False
                    break
            else:
                _cached = False
                break

        if _cached:
            logger.info(
                "All splits of the dataset are cached. Skipping preprocessing."
            )
            return

        # download all three splits from huggingface datasets
        _datasets = cast(
            DatasetDict,
            datasets.load_dataset(
                _HF_DATASET_NAME, cache_dir=str(self.manual_cache_dir)
            ),
        )
        # apply detokenizer to the text field (don't need to do this for babylm)
        # logger.info(
        #    f"Using {self.num_preprocessing_workers} workers for preprocessing datasets"
        # )
        # apply tokenizer and save raw token ids.
        if not hasattr(self.tokenizer, "encode"):
            raise ValueError("Tokenizer needs to have an encode method")

        def tokenize_fn(example):
            return {
                "text": self.tokenizer.encode(
                    example["token_ids"], add_special_tokens=False
                ),
            }

        _datasets = _datasets.map(
            tokenize_fn,
            batched=False,
            load_from_cache_file=False,  # don't save or load from cache for map
            num_proc=self.num_preprocessing_workers,
            desc="Tokenizing text",
            remove_columns=["text"],
        )
        # save the three splits in separate files
        for split, ds in _datasets.items():
            _dir = self.get_cache_dir(split)
            _dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving {split} dataset to {_dir}")
            ds.save_to_disk(_dir, num_shards=self.num_shards)

    def _apply_processors(
        self, name: str, ds: IterableDataset
    ) -> IterableDataset:
        if name in ["train/lm", "validation/lm", "test/lm"]:
            ds = ds.map(
                self.processor,
                batched=False,
                remove_columns=["token_ids"],
            )
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        return ds

    def _get_seed(self) -> int:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        return seed

    def setup(
        self, stage: Literal["fit", "validate", "test", "predict"]
    ) -> None:
        if stage == "fit" or stage is None:
            if self.train_dataset_lm is None:
                train_dataset: IterableDataset = (
                    datasets.load_from_disk(str(self.get_cache_dir("train")))
                    .to_iterable_dataset(  # type: ignore
                        num_shards=self.num_shards
                    )
                    .shuffle(buffer_size=10_000)
                )  # type: ignore
                train_dataset = self._apply_processors(
                    "train/lm", train_dataset
                )
                self.train_dataset_lm = train_dataset
            if self.val_dataset_lm is None:
                val_dataset: IterableDataset = datasets.load_from_disk(
                    str(
                        self.get_cache_dir("validation")
                    )  # following MDLM paper
                ).to_iterable_dataset(  # type: ignore
                    num_shards=self.num_shards
                )  # type: ignore
                val_dataset = self._apply_processors(
                    "validation/lm", val_dataset
                )
                self.val_dataset_lm = val_dataset
            if self.predict_dataset is None:
                self.predict_dataset = self.get_default_prediction_dataset(
                    self.tokenizer, self.block_size, self.noise_schedule
                )
        if stage == "validate":
            if self.val_dataset_lm is None:
                val_dataset: IterableDataset = datasets.load_from_disk(
                    str(self.get_cache_dir("validation"))
                ).to_iterable_dataset(  # type: ignore
                    num_shards=self.num_shards
                )
                val_dataset = self._apply_processors(
                    "validation/lm", val_dataset
                )
                self.val_dataset_lm = val_dataset
            if self.predict_dataset is None:
                self.predict_dataset = self.get_default_prediction_dataset(
                    self.tokenizer, self.block_size, self.noise_schedule
                )
        if stage == "test":
            if self.test_dataset_lm is None:
                test_dataset: IterableDataset = datasets.load_from_disk(
                    str(self.get_cache_dir("test"))
                ).to_iterable_dataset(  # type: ignore
                    num_shards=self.num_shards
                )
                test_dataset = self._apply_processors("test/lm", test_dataset)
                self.test_dataset_lm = test_dataset
            if self.predict_dataset is None:
                self.predict_dataset = self.get_default_prediction_dataset(
                    self.tokenizer, self.block_size, self.noise_schedule
                )
        if stage == "predict":
            raise NotImplementedError("No predict split provided.")

        self._setup_distributed_context()
        self._check_grad_accum()

    def _setup_distributed_context(self) -> None:
        """Called after setup() is done, when dataloaders are created."""
        # TODO: Implement _check_grad_accum here or in train_dataloader
        # check if we are using DDP
        if self._is_ddp():
            if (trainer := self.trainer) is not None:
                if self.train_dataset_lm is not None:
                    self.train_dataset_lm = split_dataset_by_node(
                        cast(IterableDataset, self.train_dataset_lm),
                        rank=trainer.global_rank,
                        world_size=trainer.world_size,
                    )
                if self.val_dataset_lm is not None:
                    self.val_dataset_lm = split_dataset_by_node(
                        cast(IterableDataset, self.val_dataset_lm),
                        rank=trainer.global_rank,
                        world_size=trainer.world_size,
                    )
            else:
                raise RuntimeError("Trainer not found")

    def train_dataloader(self) -> Any:
        if self.train_dataset_lm is not None:
            lm_dataloader = StatefulDataLoader(
                self.train_dataset_lm,  # type: ignore
                **self.train_dataloader_kwargs,
            )
        else:
            raise RuntimeError("Train dataset not found")

        return lm_dataloader

    def val_dataloader(self) -> Any:
        # TODO: Make sure that eval batch size is divisible by num_gpus_per_node so that all GPUs end up with the same number of samples
        # See the base.py
        if self.val_dataset_lm is not None:
            lm_dataloader = DataLoader(
                self.val_dataset_lm,  # type: ignore
                collate_fn=self.collator,
                **self.val_dataloader_kwargs,
            )
        else:
            raise RuntimeError("Validation dataset not found")
        # add any other dataloaders here
        # unconditional prediction dataloader
        predict_dataloader = DataLoader(
            self.predict_dataset,
            **self.val_dataloader_kwargs,
        )
        return [lm_dataloader, predict_dataloader]

    def test_dataloader(self) -> Any:
        if (
            self.test_dataset_lm is not None
            and self.test_dataset_lm is not None
        ):
            lm_dataloader = DataLoader(
                self.test_dataset_lm,  # type: ignore
                collate_fn=self.collator,
                **self.test_dataloader_kwargs,
            )
        else:
            raise RuntimeError("Test dataset not found")
        return lm_dataloader

    def predict_dataloader(self) -> DataLoader:
        pass

    @rank_zero_only
    def print_batch(
        self,
        batch: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_base(self, batch, split, dataloader_idx)


class BabylmDataModuleForIDLM(BabylmDataModule):
    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultIDLMCollatorWithDynamicPadding(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultIDLMCollatorForPredictionWithDynamicPadding(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_dataset(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> torch.utils.data.IterableDataset:
        return DefaultEmptyDataset(tokenizer, 100, empty_text="")

    @rank_zero_only
    def print_batch(
        self,
        batch: IDLMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_idlm(self, batch, split, dataloader_idx)
