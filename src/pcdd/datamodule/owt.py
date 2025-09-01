# v2

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union, cast

import datasets
import torch
from datasets import DatasetDict, IterableDataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase
from datasets import Dataset

from pcdd.datamodule.base import DataLoaderKwargs
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.os import get_num_processes
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only

from .datamodule import (
    BaseDataModule,
    Collator,
    DefaultEmptyDataset,
    DefaultIDLMCollator,
    DefaultIDLMCollatorForPrediction,
    DefaultILMCollatorForPrediction,
    DefaultILMWithLengthClassificationCollator,
    DefaultILMWithLengthClassificationCollatorForWrappedPaddedSequences,
    Tokenizer,
    ids_to_example_fn,
    print_batch_base,
    print_batch_idlm,
    print_batch_ilm,
)
from pcdd import flags
from more_itertools import chunked, flatten

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


def preprocess_fn(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Any]:
    example["token_ids"] = tokenizer.encode(  # type: ignore
        example["text"], add_special_tokens=False
    )
    return example


def chunked_iterable(iterable, block_size):
    return chunked(flatten(iterable), block_size)


def group_texts_ilm(
    examples: Dict[str, List[List[int]]],
    block_size: int,
    pad_id: int,
    type_id_extension: int = 2,
    attn_extension: bool = False,
) -> Dict[str, List[List[int]]]:
    """Convert uneven sequences into blocks of fixed size.

    1. Does not add any special tokens.
    2. Will pad the last sequence to the block size.
    """
    real_block_size = block_size
    # colapse the sequences into a single list of tokens and then create blocks of real_block_size
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for _input_ids, _attention_mask, _token_type_ids in zip(
        chunked_iterable(examples["input_ids"], real_block_size),
        chunked_iterable(examples["attention_mask"], real_block_size),
        chunked_iterable(examples["token_type_ids"], real_block_size),
    ):
        l_ = len(_input_ids)
        assert l_ == len(_attention_mask) == len(_token_type_ids)
        _input_ids += [pad_id] * (block_size - l_)
        input_ids.append(list(_input_ids))
        _attention_mask += [attn_extension] * (block_size - l_)
        attention_mask.append(list(_attention_mask))
        _token_type_ids += [type_id_extension] * (block_size - l_)
        token_type_ids.append(list(_token_type_ids))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


class OWTDataModule(BaseDataModule):
    # ref: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

    def __init__(
        self,
        manual_cache_dir: str,
        tokenizer: Tokenizer,
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
        verbosity: Literal["warning", "info", "debug"] = "info",
    ):
        super().__init__()
        # fixed parameters
        if flags.DEBUG_OWT:
            self.dataset_name = "stas/openwebtext-10k"
            self.val_split_size = 1000
        else:
            self.dataset_name = "Skylion007/openwebtext"
            self.val_split_size = 10000
        # OWT has only one split
        if flags.DEBUG_OWT:
            self.num_shards = {"train": 32, "validation": 32}
        else:
            self.num_shards = {"train": 1024, "validation": 32}
        self.manual_cache_dir: Path = (
            Path(manual_cache_dir) / self.dataset_name
        )
        # tokenizer
        self.tokenizer = cast(PreTrainedTokenizerBase, tokenizer)  # type: ignore
        self.block_size = block_size
        # datasets
        self.rewrite_manual_cache = rewrite_manual_cache
        self.train_dataset_lm = None
        self.val_dataset_lm = None
        self.test_dataset_lm = None
        self.predict_dataset_lm = None
        self.predict_dataset = None
        self.num_dataset_workers = num_dataset_workers
        self.num_preprocessing_workers = (
            num_dataset_workers or get_num_processes()
        )
        # noise schedule
        self.noise_schedule = noise_schedule
        # collators
        self.collator = collator or self.get_default_collator(
            tokenizer, block_size, noise_schedule
        )
        self.prediction_collator = (
            prediction_collator
            or self.get_default_prediction_collator(
                tokenizer, block_size, noise_schedule
            )
        )
        self.collator.tokenizer = tokenizer
        self.prediction_collator.tokenizer = tokenizer
        # dataloaders
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
        self.train_dataloder_names = {0: "lm", 1: "prediction"}
        self.val_dataloader_names = {0: "lm", 1: "prediction"}
        self.test_dataloader_names = {0: "lm", 1: "prediction"}
        self.predict_dataloader_names = {0: "prediction"}
        self.global_batch_size = global_batch_size
        # on the fly processor
        # self.processor = IdsToExampleProcessor(self.tokenizer)  # type: ignore
        datasets.utils.logging.set_verbosity(
            datasets.utils.logging.log_levels[verbosity]
        )
        self.model_type = "ilm"

    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultILMWithLengthClassificationCollatorForWrappedPaddedSequences(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultILMCollatorForPrediction(
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
            num_examples=self.val_dataloader_kwargs["batch_size"],
            empty_text="",
            tokenizer_kwargs={"return_token_type_ids": True},
        )

    def get_cache_dir(self, split: str) -> Path:
        return self.manual_cache_dir / f"{split}"

    def prepare_data(self) -> None:
        """Do global, one-time processing here.
        Note (general):
            This method is called before setup() is called on each node. There is a barrier after this method if it is called on all nodes.
            If self.prepare_data_per_node is True, this method is called on each node otherwise it is called once on the main node.
            Since this only runs on the main process, do not assign any state to self here.

        We will do the following:
        1. Download the dataset.
        2. Split the dataset into train and val.
        3. Perform raw tokenization.
        4. Save the two splits on disk.
        """
        # check if we have already cached the dataset.
        _cached = True
        for split in ["train", "validation"]:
            _dir = self.get_cache_dir(split)
            if _dir.exists():
                logger.info(
                    f"Found cached raw tokenized {split} dataset at {_dir}"
                )
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
                "All splits of the raw tokenized dataset are cached. Skipping tokenization."
            )
            do_tokenization = False
        else:
            do_tokenization = True

        # check for the existence of the final dataset
        _final_dataset_cached = False
        for split in ["train", "validation"]:
            _dir = self.get_cache_dir(split) / self.model_type
            if _dir.exists():
                logger.info(f"Found cached final {split} dataset at {_dir}")
                _final_dataset_cached = True
                break

        if _final_dataset_cached:
            return

        if do_tokenization:

            # download all three splits from huggingface datasets to hf_cache_dir
            ranked_logger.info(
                f"Downloading {self.dataset_name} dataset from huggingface datasets."
            )
            _datasets = cast(
                DatasetDict,
                datasets.load_dataset(
                    self.dataset_name,
                    num_proc=self.num_preprocessing_workers,
                    trust_remote_code=True,
                ),
            )
            # split the dataset into train and val
            # When downloaded, it has only one split "train"
            split_datasets = _datasets["train"].train_test_split(
                test_size=self.val_split_size, seed=2357, shuffle=True
            )
            split_datasets["validation"] = split_datasets.pop("test")  # rename
            # tokenize
            logger.info(
                f"Using {self.num_preprocessing_workers} workers for preprocessing datasets"
            )
            # tokenize into raw token_ids
            tokenized_datasets = split_datasets.map(
                preprocess_fn,
                batched=False,
                load_from_cache_file=False,  # don't save or load from cache for map
                num_proc=self.num_preprocessing_workers,
                desc="Tokenizing text",
                remove_columns=["text"],
                fn_kwargs={"tokenizer": self.tokenizer},
            )

            # save the two splits in separate files
            for split, ds in tokenized_datasets.items():
                _dir = self.get_cache_dir(split)
                _dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving raw tokenized {split} dataset to {_dir}")
                ds.save_to_disk(str(_dir), num_shards=self.num_shards[split])
        else:
            # load the raw tokenized datasets directly from disk
            logger.info(
                f"Loading raw tokenized datasets from {self.manual_cache_dir}"
            )
            tokenized_datasets = DatasetDict(
                {
                    "train": datasets.load_from_disk(
                        str(self.get_cache_dir("train"))
                    ),
                    "validation": datasets.load_from_disk(
                        str(self.get_cache_dir("validation"))
                    ),
                }
            )

            # save model specific grouped dataset
            tokenized_datasets["train"] = self._apply_processors(
                "train/lm", tokenized_datasets["train"]
            )
            tokenized_datasets["validation"] = self._apply_processors(
                "validation/lm", tokenized_datasets["validation"]
            )
            for split, ds in tokenized_datasets.items():
                _dir = self.get_cache_dir(split) / self.model_type
                _dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving grouped {split} dataset to {_dir}")
                ds.save_to_disk(str(_dir), num_shards=self.num_shards[split])

    def _apply_processors(self, name: str, ds: Dataset) -> Dataset:
        if name in ["train/lm", "validation/lm", "test/lm"]:
            ds = ds.map(
                ids_to_example_fn,
                batched=False,
                fn_kwargs={"tokenizer": self.tokenizer},
                remove_columns=["token_ids"],
                desc="Ids to examples",
                num_proc=self.num_preprocessing_workers,
            )  # add special tokens
            # chunk the sequences into blocks of block_size
            ds = ds.map(
                group_texts_ilm,
                batched=True,
                fn_kwargs={
                    "block_size": self.block_size,
                    "pad_id": self.tokenizer.pad_token_id,
                    "type_id_extension": 2,
                    "attn_extension": 0,
                },
                desc="Grouping text into blocks",
                num_proc=self.num_preprocessing_workers,
            )

        else:
            raise ValueError(f"Unknown dataset name: {name}")
        return ds

    def set_epoch(self, epoch: int) -> None:
        if self.train_dataset_lm is not None:
            self.train_dataset_lm.set_epoch(epoch)  # type: ignore

    def setup(
        self, stage: Literal["fit", "validate", "test", "predict"]
    ) -> None:
        if stage == "fit" or stage is None:
            if self.train_dataset_lm is None:
                train_dataset: IterableDataset = datasets.load_from_disk(
                    str(self.get_cache_dir("train") / self.model_type)
                ).to_iterable_dataset(  # type: ignore
                    num_shards=self.num_shards["train"]
                )  # type: ignore
                train_dataset = train_dataset.shuffle(
                    buffer_size=10_000, seed=42
                )
                self.train_dataset_lm = train_dataset
            if self.val_dataset_lm is None:
                val_dataset: IterableDataset = datasets.load_from_disk(
                    str(self.get_cache_dir("validation") / self.model_type)
                ).to_iterable_dataset(  # type: ignore
                    num_shards=self.num_shards["validation"]
                )  # type: ignore
                self.val_dataset_lm = val_dataset
            if self.predict_dataset is None:
                self.predict_dataset = self.get_default_prediction_dataset(
                    self.tokenizer, self.block_size, self.noise_schedule
                )
        if stage == "validate":
            if self.val_dataset_lm is None:
                val_dataset: IterableDataset = datasets.load_from_disk(
                    str(self.get_cache_dir("validation") / self.model_type)
                ).to_iterable_dataset(  # type: ignore
                    num_shards=self.num_shards["validation"]
                )
                self.val_dataset_lm = val_dataset
            if self.predict_dataset is None:
                self.predict_dataset = self.get_default_prediction_dataset(
                    self.tokenizer, self.block_size, self.noise_schedule
                )
        if stage == "test":
            if self.test_dataset_lm is None:
                test_dataset: IterableDataset = datasets.load_from_disk(
                    str(self.get_cache_dir("validation") / self.model_type)
                ).to_iterable_dataset(  # type: ignore
                    num_shards=self.num_shards["validation"]
                )
                self.test_dataset_lm = test_dataset
            if self.predict_dataset is None:
                self.predict_dataset = self.get_default_prediction_dataset(
                    self.tokenizer, self.block_size, self.noise_schedule
                )
        if stage == "predict":
            if self.predict_dataset is None:
                self.predict_dataset = self.get_default_prediction_dataset(
                    self.tokenizer, self.block_size, self.noise_schedule
                )

        self._setup_distributed_context()
        self._check_grad_accum()

    def _setup_distributed_context(self) -> None:
        """Called after setup() is done, when dataloaders are created."""
        if self._is_ddp():
            if (trainer := self.trainer) is not None:
                if self.train_dataset_lm is not None:
                    self.train_dataset_lm = split_dataset_by_node(  # type: ignore
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
                if self.test_dataset_lm is not None:
                    self.test_dataset_lm = split_dataset_by_node(
                        cast(IterableDataset, self.test_dataset_lm),
                        rank=trainer.global_rank,
                        world_size=trainer.world_size,
                    )
            else:
                raise RuntimeError("Trainer not found")

    def train_dataloader(self) -> Any:
        if self.train_dataset_lm is not None:
            lm_dataloader = StatefulDataLoader(
                self.train_dataset_lm,  # type: ignore
                collate_fn=self.collator,
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
        predict_dataloader = DataLoader(
            self.predict_dataset,
            collate_fn=self.prediction_collator,
            **{
                **self.val_dataloader_kwargs,
                "persistent_workers": False,
                "prefetch_factor": None,
                "num_workers": 0,
            },
        )
        return [lm_dataloader, predict_dataloader]

    def test_dataloader(self) -> Any:
        if (
            self.test_dataset_lm is not None
            and self.test_dataloader_kwargs is not None
        ):
            lm_dataloader = DataLoader(
                self.test_dataset_lm,  # type: ignore
                collate_fn=self.collator,
                **self.test_dataloader_kwargs,
            )
        else:
            raise RuntimeError("Test dataset not found")
        return lm_dataloader

    def predict_dataloader(self) -> Any:
        if self.predict_dataset is not None:
            return DataLoader(
                self.predict_dataset,
                collate_fn=self.prediction_collator,
                **self.predict_dataloader_kwargs,
            )
        else:
            raise RuntimeError("Predict dataset not found")

    @rank_zero_only
    def print_batch(
        self,
        batch: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ) -> None:
        print_batch_ilm(self, batch, split, dataloader_idx)


OWTForILMStoppingDataModule = OWTDataModule
