# v2

import re
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    TypedDict,
    Union,
    cast,
)

import datasets
import torch
from datasets import DatasetDict, IterableDataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from pcdd.datamodule.base import DataLoaderKwargs
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.os import get_num_processes
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only

from .datamodule import (
    BaseBatch,
    BaseCollatorInput,
    BaseDataModule,
    Collator,
    DefaultEmptyDataset,
    DefaultIDLMCollator,
    DefaultIDLMCollatorForPrediction,
    DefaultILMCollatorForPrediction,
    DefaultILMWithLengthClassificationCollator,
    ILMBatch,
    Tokenizer,
    ids_to_example_fn,
    pad_dynamic,
    print_batch_base,
    print_batch_idlm,
    print_batch_ilm,
)

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


class ANLGRawExample(TypedDict):
    """Raw example from allenai/art dataset."""

    observation_1: str
    observation_2: str
    hypothesis_1: str
    hypothesis_2: str
    label: int


class PreProcessedANLGExample(TypedDict):
    """Pre-processed example for ANLG."""

    observation_1_token_ids: List[int]
    observation_2_token_ids: List[int]
    hypothesis_1_token_ids: List[int]
    hypothesis_2_token_ids: List[int]
    label: int


def ids_to_example_fn_for_ilm_stopping(
    example: PreProcessedANLGExample,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[
    Literal["input_ids", "attention_mask", "token_type_ids", "constraint"],
    List[int],
]:
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
        example["observation_2_token_ids"],
        example["observation_1_token_ids"],
    )
    attention_mask = [1] * len(input_ids)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(
        example["observation_2_token_ids"],
        example["observation_1_token_ids"],
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def preprocess_fn(
    example: ANLGRawExample, tokenizer: PreTrainedTokenizerBase
) -> PreProcessedANLGExample:
    observation_1 = tokenizer.encode(
        example["observation_1"], add_special_tokens=False
    )
    observation_2 = tokenizer.encode(
        example["observation_2"], add_special_tokens=False
    )
    hypothesis_1 = tokenizer.encode(
        example["hypothesis_1"], add_special_tokens=False
    )
    hypothesis_2 = tokenizer.encode(
        example["hypothesis_2"], add_special_tokens=False
    )
    label = example["label"]
    return {
        "observation_1_token_ids": observation_1,
        "observation_2_token_ids": observation_2,
        "hypothesis_1_token_ids": hypothesis_1,
        "hypothesis_2_token_ids": hypothesis_2,
        "label": label,
    }


def prediction_collator_ilm_stopping(
    examples: List[BaseCollatorInput],
    tokenizer: PreTrainedTokenizerBase,
) -> ILMBatch:
    padded_examples: BaseBatch = pad_dynamic(
        examples,
        pad_token_id=tokenizer.pad_token_id,
        attn_extension=0,
        type_extension=2,
    )
    constraint = torch.ones_like(
        padded_examples["input_ids"], dtype=torch.bool
    )
    # set the first position where type_ids == 2 to False
    last_indices_of_1 = (padded_examples["token_type_ids"] <= 1).sum(
        dim=-1
    ) - 1
    constraint[torch.arange(constraint.shape[0]), last_indices_of_1] = False
    return {
        "input_ids": padded_examples["input_ids"],
        "attention_mask": padded_examples["attention_mask"],
        "token_type_ids": padded_examples["token_type_ids"],
        "drop": torch.zeros_like(
            padded_examples["input_ids"], dtype=torch.bool
        ),
        "target_ids": None,  # type: ignore
        "constraint": constraint,
    }


class PredictionCollatorILMStopping:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        return prediction_collator_ilm_stopping(examples, self.tokenizer)


class ANLGDataModule(BaseDataModule):

    def __init__(
        self,
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        noise_schedule: NoiseSchedule,
        # train_dataloader_kwargs: DataLoaderKwargs,
        # val_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        # test_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        predict_dataloader_kwargs: DataLoaderKwargs,
        rewrite_manual_cache: bool = False,
        block_size: int = 128,
        global_batch_size: int = 512,
        num_dataset_workers: Optional[int] = None,
        # collator: Optional[Collator] = None,
        prediction_collator: Optional[Collator] = None,
        verbosity: Literal["warning", "info", "debug"] = "info",
        **kwargs,  # need this to pass some kwargs
    ):
        super().__init__()
        # fixed parameters
        self.dataset_name = "allenai/art"
        self.train_split = "train"
        self.val_split = "validation"
        self.test_split = "validation"
        self.predict_split = "validation"
        self.num_shards = 128
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
        # TODO (prediction only)
        # self.collator = collator or self.get_default_collator(
        #    tokenizer, block_size, noise_schedule
        # )
        self.prediction_collator = (
            prediction_collator
            or self.get_default_prediction_collator(
                tokenizer, block_size, noise_schedule
            )
        )
        self.ids_to_example_fn = self.get_default_ids_to_example_fn()
        # TODO (prediction only)
        # self.collator.tokenizer = tokenizer
        self.prediction_collator.tokenizer = tokenizer
        # dataloaders
        # TODO (prediction only)
        # self.train_dataloader_kwargs = train_dataloader_kwargs
        # self.val_dataloader_kwargs = val_dataloader_kwargs or {
        #    **deepcopy(train_dataloader_kwargs),
        #    "shuffle": False,
        # }
        # self.test_dataloader_kwargs = test_dataloader_kwargs or {
        #    **deepcopy(train_dataloader_kwargs),
        #    "shuffle": False,
        # }
        self.predict_dataloader_kwargs = predict_dataloader_kwargs
        # self.train_dataloder_names = {0: "lm", 1: "prediction"}
        # self.val_dataloader_names = {0: "lm", 1: "prediction"}
        # self.test_dataloader_names = {0: "lm", 1: "prediction"}
        self.predict_dataloader_names = {0: "prediction"}
        self.global_batch_size = global_batch_size
        # on the fly processor
        # self.processor = IdsToExampleProcessor(self.tokenizer)  # type: ignore
        datasets.utils.logging.set_verbosity(
            datasets.utils.logging.log_levels[verbosity]
        )

    def get_default_prediction_collator(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> PredictionCollatorILMStopping:
        return PredictionCollatorILMStopping(tokenizer)  # type: ignore

    def get_default_ids_to_example_fn(self) -> Callable:
        return ids_to_example_fn_for_ilm_stopping

    def get_cache_dir(self, split: str) -> Path:
        return self.manual_cache_dir / f"{split}"

    def prepare_data(self) -> None:
        """Do global, one-time processing here.
        Note (general):
            This method is called before setup() is called on each node. There is a barrier after this method if it is called on all nodes.
            If self.prepare_data_per_node is True, this method is called on each node otherwise it is called once on the main node.
            Since this only runs on the main process, do not assign any state to self here.

        We will do the following:
        1. Download from https://huggingface.co/datasets/allenai/art
        2. Perform tokenization and store raw tokens to the disk.
        """
        # check if we have already cached the dataset.
        _cached = True
        for split in [
            "train",
            "validation",
        ]:  # allenai/art has only two splits
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
            desc="Detokenizing and tokenizing text",
            fn_kwargs={"tokenizer": self.tokenizer},
        )
        # make sure that the keys in the dataset dict match our splits

        # save the three splits in separate files
        for split, ds in _datasets.items():
            _dir = self.get_cache_dir(split)
            _dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving {split} dataset to {_dir}")
            ds.save_to_disk(_dir, num_shards=self.num_shards)

    def setup(
        self, stage: Literal["fit", "validate", "test", "predict"]
    ) -> None:
        if stage != "predict":
            raise NotImplementedError(
                "For now we only support prediciton on ANLG dataset."
            )
        if stage == "predict":
            if self.predict_dataset is None:
                self.predict_dataset = datasets.load_from_disk(
                    str(self.get_cache_dir(self.predict_split))
                )
                self.predict_dataset = self.predict_dataset.map(
                    self.ids_to_example_fn,
                    batched=False,
                    load_from_cache_file=False,
                    num_proc=self.num_preprocessing_workers,
                    fn_kwargs={"tokenizer": self.tokenizer},
                    desc="Converting to model format",
                )

        if self._is_ddp():
            raise NotImplementedError("DDP is not supported for ANLG")

    def predict_dataloader(self) -> Any:
        if self.predict_dataset is None:
            raise ValueError("Predict dataset is not set. ")
        predict_dataloader = DataLoader(
            self.predict_dataset,  # type: ignore
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
        batch: ILMBatch,
        split: Literal["train", "val", "test", "predict"],
        dataloader_idx: Optional[int] = None,
    ) -> None:
        print_batch_ilm(self, batch, split, dataloader_idx)


ANLGForILMStoppingDataModule = ANLGDataModule
