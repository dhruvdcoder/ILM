# v2
"""Very similar to the LM1BDataModule, but for combined tiny_stories and roc_stories dataset."""
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import datasets
import torch
from datasets import DatasetDict, IterableDataset, Dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
import lightning as L

from pcdd import flags
from pcdd.datamodule.base import DataLoaderKwargs
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.os import get_num_processes
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only

from .datamodule import (
    BaseCollatorInput,
    BaseDataModule,
    Collator,
    DefaultEmptyDataset,
    DefaultIDLMCollator,
    DefaultIDLMCollatorForPrediction,
    DefaultILMCollatorForPrediction,
    DefaultILMWithLengthClassificationCollator,
    DefaultMDLMCollator,
    DefaultMDLMCollatorForPrediction,
    DefaultARLMCollator,
    DefaultARLMCollatorForPrediction,
    ILMBatch,
    MDLMEmptyDataset,
    ARLMEmptyDataset,
    Tokenizer,
    ids_to_example_fn,
    pad_truncate,
    print_batch_base,
    print_batch_idlm,
    print_batch_ilm,
    print_batch_mdlm,
    print_batch_arlm,
)

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


def preprocess_fn(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Any]:
    example["token_ids"] = tokenizer.encode(  # type: ignore
        example["text"], add_special_tokens=False
    )
    return example


class StoriesDataModule(BaseDataModule):

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
        predict_dataset: Optional[IterableDataset] = None,
        num_unconditional_samples: Optional[int] = None,
    ):
        super().__init__()
        # fixed parameters
        self.dataset_name = "dhruveshpatel/tiny_roc_stories"
        self.train_split = "train"
        self.val_split = "validation"
        self.test_split = "validation"
        self.num_shards = 256
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
        self.predict_dataset = predict_dataset
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
        self.num_unconditional_samples = num_unconditional_samples
        # on the fly processor
        # self.processor = IdsToExampleProcessor(self.tokenizer)  # type: ignore
        datasets.utils.logging.set_verbosity(
            datasets.utils.logging.log_levels[verbosity]
        )

    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("No default collator for LM1B")

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("No default prediction collator for LM1B")

    def get_default_prediction_dataset(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> torch.utils.data.IterableDataset:
        raise NotImplementedError("No default prediction dataset for LM1B")

    def get_cache_dir(self, split: str) -> Path:
        return self.manual_cache_dir / f"{split}"

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
        for split in ["train", "validation"]:
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

        # download all three splits from huggingface datasets to hf_cache_dir
        ranked_logger.info(
            f"Downloading {self.dataset_name} dataset from huggingface datasets."
        )
        _datasets = cast(
            DatasetDict,
            datasets.load_dataset(self.dataset_name),
        )
        # apply detokenizer to the text field
        logger.info(
            f"Using {self.num_preprocessing_workers} workers for preprocessing datasets"
        )

        _datasets = _datasets.map(
            preprocess_fn,
            batched=False,
            load_from_cache_file=False,  # don't save or load from cache for map
            num_proc=self.num_preprocessing_workers,
            desc="tokenizing text",
            remove_columns=["text"],
            fn_kwargs={"tokenizer": self.tokenizer},
        )
        # make sure that the keys in the dataset dict match our splits

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
            # don't explicitly set num_proc here.
            ds = ds.map(
                ids_to_example_fn,
                batched=False,
                fn_kwargs={"tokenizer": self.tokenizer},
                remove_columns=["source"],
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
                    str(self.get_cache_dir(self.train_split))
                ).to_iterable_dataset(  # type: ignore
                    num_shards=self.num_shards
                )  # type: ignore
                if not flags.DEBUG_OVERFIT:
                    train_dataset = train_dataset.shuffle(
                        buffer_size=10_000, seed=42
                    )

                train_dataset = self._apply_processors(
                    "train/lm", train_dataset
                )
                self.train_dataset_lm = train_dataset
            if self.val_dataset_lm is None:
                val_dataset: IterableDataset = datasets.load_from_disk(
                    str(
                        self.get_cache_dir(self.val_split)
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
                    str(
                        self.get_cache_dir(self.val_split)
                    )  # following MDLM paper
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
                    str(self.get_cache_dir(self.test_split))
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
            if self.predict_dataset is None:
                self.predict_dataset = self.get_default_prediction_dataset(
                    self.tokenizer, self.block_size, self.noise_schedule
                )
        self._setup_distributed_context()
        self._check_grad_accum()

    def _setup_distributed_context(self) -> None:
        """Called after setup() is done, when dataloaders are created."""
        # TODO: Implement _check_grad_accum here or in train_dataloader
        # check if we are using DDP
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
        predict_dataloader = DataLoader(
            self.predict_dataset,
            collate_fn=self.prediction_collator,
            **{
                **self.predict_dataloader_kwargs,
                "persistent_workers": False,
                "prefetch_factor": None,
                "num_workers": 0,
            },
        )
        return predict_dataloader

    @rank_zero_only
    def print_batch(
        self,
        batch: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ) -> None:
        print_batch_base(self, batch, split, dataloader_idx)


class StoriesForILMStoppingDataModule(StoriesDataModule):
    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultILMWithLengthClassificationCollator(
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
        num_examples = (
            self.num_unconditional_samples
            or self.predict_dataloader_kwargs["batch_size"]
        )
        return DefaultEmptyDataset(
            tokenizer,
            num_examples=num_examples,
            empty_text="",
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ) -> None:
        print_batch_ilm(self, batch, split, dataloader_idx)


class StoriesForIDLMDataModule(StoriesDataModule):
    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultIDLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultIDLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_dataset(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> torch.utils.data.IterableDataset:
        num_examples = (
            self.num_unconditional_samples
            or self.predict_dataloader_kwargs["batch_size"]
        )
        return DefaultEmptyDataset(
            tokenizer,
            num_examples=num_examples,
            empty_text="",
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ) -> None:
        print_batch_idlm(self, batch, split, dataloader_idx)


class StoriesForARLMDataModule(StoriesDataModule):
    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultARLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultARLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_dataset(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> torch.utils.data.IterableDataset:
        return ARLMEmptyDataset(
            num_examples=self.predict_dataloader_kwargs["batch_size"],
            tokenizer=tokenizer,
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ) -> None:
        print_batch_arlm(self, batch, split, dataloader_idx)


class ILMWithConstraintCollator(DefaultILMWithLengthClassificationCollator):
    """Prepares constraint using the drop tokens."""

    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        batch["constraint"] = torch.ones_like(batch["drop"]).logical_and_(
            torch.roll(batch["drop"], shifts=-1, dims=-1).logical_not_()
        )
        return batch


class ILMWithConstraintCollatorForPrediction(DefaultILMCollatorForPrediction):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        batch["constraint"] = torch.ones_like(batch["drop"]).logical_and_(
            torch.roll(batch["drop"], shifts=-1, dims=-1).logical_not_()
        )
        # for unconditional prediction the rightmost token will be BOS
        batch["constraint"][:, -1] = False
        return batch


class StoriesForILMStoppingWithConstraintDataModule(StoriesDataModule):
    """Sends meaningful constraints in the input."""

    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ILMWithConstraintCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return ILMWithConstraintCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_dataset(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> torch.utils.data.IterableDataset:
        num_examples = (
            self.num_unconditional_samples
            or self.predict_dataloader_kwargs["batch_size"]
        )
        return DefaultEmptyDataset(
            tokenizer,
            num_examples=num_examples,
            empty_text="",
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ) -> None:
        print_batch_ilm(self, batch, split, dataloader_idx)


class StoriesForMDLMDataModule(StoriesDataModule):
    def get_default_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMDLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMDLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_dataset(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> torch.utils.data.IterableDataset:
        num_examples = (
            self.num_unconditional_samples
            or self.predict_dataloader_kwargs["batch_size"]
        )
        return MDLMEmptyDataset(
            num_examples=num_examples,
            max_length=block_size,
            tokenizer=tokenizer,
        )


########################################################
# region: Infilling


def ilm_infilling_processing_fn(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    token_ids = tokenizer.convert_tokens_to_ids(example["tokenized_text"])
    infill_mask = example["mask"]
    if len(infill_mask) != len(token_ids):
        raise ValueError(
            f"Infill mask length {len(infill_mask)} does not match tokenized text length {len(token_ids)}"
        )
    next_is_infill = infill_mask[1:]
    constraint = []
    for i in range(len(token_ids) - 1):
        if next_is_infill[i]:
            constraint.append(False)
        else:
            constraint.append(True)
    constraint.append(True)
    input_ids = []
    final_constraint = []
    for input_id, infill, c in zip(token_ids, infill_mask, constraint):
        if not infill:
            input_ids.append(input_id)
            final_constraint.append(c)

    # create attention mask and token type ids
    _example = ids_to_example_fn(
        {"token_ids": input_ids},  # type: ignore
        tokenizer,
        block_size=None,
    )
    #
    if infill_mask[0] == 1:
        constraint_bos = False
    else:
        constraint_bos = True
    example["input_ids"] = _example["input_ids"]
    example["attention_mask"] = _example["attention_mask"]
    example["token_type_ids"] = _example["token_type_ids"]
    example["drop"] = [False] * len(_example["input_ids"])
    example["input_token_ids"] = [
        tokenizer.cls_token_id,
        tokenizer.bos_token_id,
    ] + token_ids
    example["input_infill_mask"] = [0, 0] + infill_mask
    example["constraint"] = [True, constraint_bos] + final_constraint
    example["target_ids"] = None
    if len(example["input_ids"]) != len(example["constraint"]):
        raise ValueError(
            f"Input ids length {len(example['input_ids'])} does not match constraint length {len(example['constraint'])}"
        )
    return example


def ilm_infilling_prediction_collator(
    examples: List[Dict],  # each dict is an example
    tokenizer: PreTrainedTokenizerBase,
) -> Dict:  # batch
    # keys: ['text', 'tokenized_text', 'mask', 'processed_text', 'input_ids', 'attention_mask', 'token_type_ids', 'drop', 'input_token_ids', 'input_infill_mask', 'constraint', 'target_ids']
    max_len = max(len(e["input_ids"]) for e in examples)
    max_original_input_len = max(len(e["input_token_ids"]) for e in examples)
    pad_token_id = tokenizer.pad_token_id
    type_extension = 2
    attn_extension = 0
    batch = {
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
        "constraint": torch.tensor(
            [
                example["constraint"][:max_len]
                + [1] * max(0, max_len - len(example["constraint"]))
                for example in examples
            ],
            dtype=torch.bool,
        ),
        "target_ids": None,  # type: ignore
        "input_token_ids": torch.tensor(
            [
                example["input_token_ids"][:max_original_input_len]
                + [pad_token_id]
                * max(
                    0, max_original_input_len - len(example["input_token_ids"])
                )
                for example in examples
            ],
            dtype=torch.long,
        ),
        "input_infill_mask": torch.tensor(
            [
                example["input_infill_mask"][:max_original_input_len]
                + [0]
                * max(
                    0,
                    max_original_input_len - len(example["input_infill_mask"]),
                )
                for example in examples
            ],
            dtype=torch.bool,
        ),
    }
    batch["drop"] = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
    masked_input_ids = batch["input_token_ids"].masked_fill(
        batch["input_infill_mask"], tokenizer.mask_token_id
    )
    batch["masked_input_ids"] = torch.tensor(
        masked_input_ids,
        dtype=torch.long,
    )
    return batch


class StoriesForILMStoppingInfillingDataModule(BaseDataModule):
    prediction_datasets = {
        "train_tiny_stories_sentences_1": "AvinashAmballa/TinyStories_Train_1000_Masked-SingleSentence",
        "tiny_stories_sentences_1": "AvinashAmballa/TinyStories_Masked-SingleSentence",
        "tiny_stories_sentences_50": "AvinashAmballa/TinyStories_Masked-50",
        "tiny_stories_sentences_20": "AvinashAmballa/TinyStories_Masked-20",
        "roc_stories_sentences_50": "AvinashAmballa/rocStories_Masked-50",
        "roc_stories_sentences_20": "AvinashAmballa/rocStories_Masked-20",
    }

    def __init__(
        self,
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        *args,
        infill_dataset_name: Literal[
            "roc_stories", "tiny_stories", "train_tiny_stories"
        ] = "tiny_stories",
        infill_percent: int = 20,
        infill_type: Literal["sentences", "words"] = "sentences",
        num_dataset_workers: Optional[int] = None,
        predict_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        loss_on_padding: bool = True,
        block_size: int = 1024,
        num_examples: int = 4000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_schedule = noise_schedule
        self.block_size = block_size
        self.infill_dataset_name = infill_dataset_name
        self.infill_percent = infill_percent
        self.tokenizer = tokenizer
        self.infill_type = infill_type
        self.num_dataset_workers = num_dataset_workers
        self.predict_dataloader_kwargs = (
            predict_dataloader_kwargs
            if predict_dataloader_kwargs is not None
            else {
                "batch_size": 1,
                "num_workers": 0,
                "persistent_workers": False,
                "prefetch_factor": None,
            }
        )
        self.prepare_data_per_node = False
        self.predict_dataloader_names = {0: "infill"}
        self.loss_on_padding = loss_on_padding
        self.num_examples = num_examples

    def prepare_data(self) -> None:
        logger.info("Nothing to prepare for infilling datasets")

    def setup(
        self, stage: Literal["fit", "validate", "test", "predict"]
    ) -> None:
        logger.info("Nothing to setup for infilling datasets")

    def get_prediction_dataset(
        self,
        tokenizer: Tokenizer,
    ) -> IterableDataset:
        dataset_name = self.prediction_datasets[
            f"{self.infill_dataset_name}_{self.infill_type}_{self.infill_percent}"
        ]
        if "tiny" in dataset_name.lower():
            if "train" in dataset_name.lower():
                split = "train"
            else:
                split = "validation"
        else:
            split = "test"
        logger.info(f"Loading prediction dataset {dataset_name}/{split}")
        ds = datasets.load_dataset(dataset_name, split=split)
        ds = ds.select(range(self.num_examples))
        ds = ds.map(
            ilm_infilling_processing_fn,
            fn_kwargs={"tokenizer": tokenizer},
            batched=False,
            num_proc=self.num_dataset_workers,
            load_from_cache_file=False,
        )
        return ds

    def train_dataloader(self) -> Any:
        raise NotImplementedError("Only prediction dataloader is implemented")

    def val_dataloader(self) -> Any:
        raise NotImplementedError("Only prediction dataloader is implemented")

    def test_dataloader(self) -> Any:
        raise NotImplementedError("Only prediction dataloader is implemented")

    def predict_dataloader(self) -> Any:
        kwargs = deepcopy(self.predict_dataloader_kwargs)
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = None
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        dataset = self.get_prediction_dataset(self.tokenizer)
        return DataLoader(
            dataset,
            collate_fn=partial(
                ilm_infilling_prediction_collator, tokenizer=self.tokenizer
            ),
            **kwargs,
        )

    def print_batch(
        self,
        batch,
        split: Literal["train", "val", "test", "predict"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_ilm(self, batch, split, dataloader_idx)


def mdlm_infilling_processing_fn(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    token_ids = tokenizer.convert_tokens_to_ids(example["tokenized_text"])
    infill_mask = example["mask"]
    if len(infill_mask) != len(token_ids):
        raise ValueError(
            f"Infill mask length {len(infill_mask)} does not match tokenized text length {len(token_ids)}"
        )
    # create attention mask and token type ids
    _example = ids_to_example_fn(
        {"token_ids": token_ids},  # type: ignore
        tokenizer,
        block_size=None,
    )
    diff = len(_example["input_ids"]) - len(infill_mask)
    if diff == 1:
        drop = [0] + infill_mask
    elif diff == 2:
        drop = [0] + infill_mask + [0]
    else:
        raise ValueError(
            f"Difference between input_ids and infill_mask is {diff}, which is not supported"
        )
    return {
        "input_ids": _example["input_ids"],
        "attention_mask": _example["attention_mask"],
        "token_type_ids": _example["token_type_ids"],
        "drop": drop,
    }


def mdlm_infilling_prediction_collator(
    examples: List[Dict],  # each dict is an example
    tokenizer: PreTrainedTokenizerBase,
    noise_schedule: NoiseSchedule,
    block_size: int,
    loss_on_padding: bool = True,
) -> Dict:  # batch
    if not loss_on_padding:
        raise NotImplementedError("Loss on padding must be True")
    max_seq_len = max(len(ex["input_ids"]) for ex in examples)
    batch_size = len(examples)
    # TODO: estimate t from number of tokens masked
    t = torch.ones((batch_size,), dtype=torch.float32)
    noise_rate, total_noise = noise_schedule(t)
    # from pad_truncate
    max_len = max_seq_len
    pad_token_id = tokenizer.pad_token_id
    attn_extension = 0
    type_extension = 2
    batch = {
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
        "drop": torch.tensor(
            [
                example["drop"][:max_len]
                + [0] * max(0, max_len - len(example["drop"]))
                for example in examples
            ],
            dtype=torch.bool,
        ),
        "t": t,
        "noise_rate": noise_rate,
        "total_noise": total_noise,
        "constraint": None,
    }
    # apply the mask to the input_ids
    batch["input_ids"] = batch["input_ids"].masked_fill(
        batch["drop"], tokenizer.mask_token_id
    )
    batch["original_input_ids"] = torch.tensor(
        [example["input_ids"] for example in examples],
        dtype=torch.long,
    )
    return batch


class StoriesForMDLMInfillingDataModule(
    StoriesForILMStoppingInfillingDataModule
):
    def get_prediction_dataset(
        self,
        tokenizer: Tokenizer,
    ) -> Dataset:
        dataset_name = self.prediction_datasets[
            f"{self.infill_dataset_name}_{self.infill_type}_{self.infill_percent}"
        ]
        if "tiny" in dataset_name.lower():
            split = "validation"
        else:
            split = "test"
        logger.info(f"Loading prediction dataset {dataset_name}/{split}")
        ds = datasets.load_dataset(dataset_name, split=split)
        ds = ds.select(range(self.num_examples))
        ds = ds.map(
            mdlm_infilling_processing_fn,
            fn_kwargs={"tokenizer": tokenizer},
            batched=False,
            num_proc=self.num_dataset_workers,
            load_from_cache_file=False,
        )
        return ds

    def predict_dataloader(self) -> Any:
        dataset = self.get_prediction_dataset(self.tokenizer)
        return DataLoader(
            dataset,
            collate_fn=partial(
                mdlm_infilling_prediction_collator,
                tokenizer=self.tokenizer,
                noise_schedule=self.noise_schedule,
                block_size=self.block_size,
                loss_on_padding=self.loss_on_padding,
            ),
            **self.predict_dataloader_kwargs,
        )

    def print_batch(
        self,
        batch,
        split: Literal["train", "val", "test", "predict"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_mdlm(self, batch, split, dataloader_idx)


# endregion: Infilling
########################################################
