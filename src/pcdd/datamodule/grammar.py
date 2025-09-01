from pathlib import Path
import random
from typing import List, Literal
from nltk import CFG as _CFG

from nltk.parse.generate import generate
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets.distributed import split_dataset_by_node
from pcdd import flags
from pcdd.datamodule.base import DataLoaderKwargs
from pcdd.datamodule.datamodule import (
    ARLMTokenizerMixin,
    BaseDataModule,
    Collator,
    DefaultEmptyDataset,
    IDLMTokenizerMixin,
    MDLMTokenizerMixin,
    Tokenizer,
    ids_to_example_fn,
    DefaultIDLMCollator,
    DefaultIDLMCollatorForPrediction,
    DefaultILMCollator,
    DefaultILMCollatorForPrediction,
    DefaultMDLMCollator,
    DefaultMDLMCollatorForPrediction,
    DefaultARLMCollator,
    DefaultARLMCollatorForPrediction,
)
from transformers import PreTrainedTokenizer, AddedToken
from typing import Dict, Optional, Union
from nltk.parse import EarleyChartParser
import hashlib
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only
import datasets
from pcdd.datamodule.datamodule import (
    print_batch_idlm,
    print_batch_ilm,
    print_batch_mdlm,
    print_batch_arlm,
)

logger = RankedLogger(__name__, rank_zero_only=True)


class CFG(_CFG):
    def get_vocab(self) -> List[str]:
        terminals = set()
        for production in self.productions():
            for symbol in production.rhs():
                if isinstance(symbol, str):  # Terminals are strings
                    terminals.add(symbol)
        return list(terminals)


class SpaceTokenizer(PreTrainedTokenizer, Tokenizer):
    model_input_names: List[str] = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: List[str],
        parser_args: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Initializes the SpaceTokenizer with a vocabulary and optional parser arguments."""
        self.vocab = vocab
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
        self.offset = 7

        super().__init__(
            eos_token=eos_token,
            cls_token=cls_token,
            pad_token=pad_token,
            bos_token=bos_token,
            mask_token=mask_token,
            sep_token=sep_token,
            unk_token=unk_token,
            add_prefix_space=False,
            **kwargs,
        )

    def __len__(self):
        return len(self._vocab_str_to_int)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        # just split on space
        # ensure all tokens are in vocab
        tokens = []
        for token in text.split(" "):
            if not token:  # skip empty strings
                continue
            if token not in self._vocab_str_to_int:
                raise ValueError(f"Token {token} not in vocab")
            tokens.append(token)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int[
            token
        ]  # let it throw KeyError if token is not in vocab

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
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
            # Remove BOS and CLS for parsability
            to_skip = {
                self.mask_token,
                self.pad_token,
                self.bos_token,
                self.cls_token,
            }
            tokens = [t for t in tokens if t not in to_skip]
        return " ".join(tokens)


class IDLMSpaceTokenizer(IDLMTokenizerMixin, SpaceTokenizer):
    pass


class ILMSpaceTokenizer(IDLMTokenizerMixin, SpaceTokenizer):
    pass


class MDLMSpaceTokenizer(MDLMTokenizerMixin, SpaceTokenizer):
    pass


class ARLMSpaceTokenizer(ARLMTokenizerMixin, SpaceTokenizer):
    pass


def preprocess_fn(example, tokenizer):
    # Tokenize and convert to ids
    text = example["text"]
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return {"token_ids": token_ids}


# Global preprocessing function for CFGDataModule
def cfg_preprocess_fn(example, tokenizer):
    # Tokenize and convert to ids
    text = example["text"]
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids_to_example_fn({"token_ids": token_ids}, tokenizer)


class CFGDataModule(BaseDataModule):
    prepare_data_per_node: bool = False

    def get_tokenizer(self, vocab: List[str]):
        raise NotImplementedError("Not implemented")

    def get_default_prediction_dataset(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> torch.utils.data.IterableDataset:
        raise NotImplementedError("Not implemented")

    def get_default_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("Not implemented")

    def get_default_prediction_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("Not implemented")

    @staticmethod
    def _get_default_dataloader_kwargs(
        type: Literal["train", "val", "test", "predict"],
    ) -> DataLoaderKwargs:
        return {
            "batch_size": 64,
            "num_workers": 4,
            "shuffle": True if type == "train" else False,
            "pin_memory": True,
        }

    @staticmethod
    def hash_cfg(cfg, alg="sha256"):
        """
        Compute a content-based hash for an nltk.CFG instance.

        Parameters
        ----------
        cfg : nltk.CFG
            The grammar to hash.
        alg : str
            Any algorithm supported by hashlib (e.g. "md5", "sha1", "sha256").

        Returns
        -------
        hexdigest : str
            The hexadecimal digest of the hash.
        """
        # 1. Canonicalize productions
        prods = sorted(str(p) for p in cfg.productions())
        # 2. Build a single byte string
        blob = "\n".join([f"START→{cfg.start()}"] + prods).encode("utf‑8")
        # 3. Hash
        h = hashlib.new(alg, blob)
        return h.hexdigest()

    def __init__(
        self,
        manual_cache_dir: str,
        grammar: CFG,
        noise_schedule: NoiseSchedule,
        parser_args: Optional[Dict[str, str]] = None,
        max_depth: int = 20,
        train_size: float = 0.6,
        val_size: float = 0.1,
        test_size: float = 0.3,
        rewrite_manual_cache: bool = False,
        train_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        val_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        test_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        predict_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        block_size: int = 128,
        global_batch_size: int = 64,
        num_dataset_workers: Optional[int] = None,
        num_unconditional_samples: Optional[int] = None,
    ):
        """Initializes the CFGDataModule for generating, caching, and loading all possible sequences from a CFG grammar."""
        super().__init__()
        self.grammar = grammar
        self.parser = EarleyChartParser(grammar, **(parser_args or {}))
        self.tokenizer = self.get_tokenizer(vocab=grammar.get_vocab())
        self.manual_cache_dir = manual_cache_dir
        self.rewrite_manual_cache = rewrite_manual_cache
        self.noise_schedule = noise_schedule
        self.train_dataloader_kwargs = (
            train_dataloader_kwargs
            if train_dataloader_kwargs is not None
            else self._get_default_dataloader_kwargs("train")
        )
        self.val_dataloader_kwargs = (
            val_dataloader_kwargs
            if val_dataloader_kwargs is not None
            else self._get_default_dataloader_kwargs("val")
        )
        self.test_dataloader_kwargs = (
            test_dataloader_kwargs
            if test_dataloader_kwargs is not None
            else self._get_default_dataloader_kwargs("test")
        )
        self.predict_dataloader_kwargs = (
            predict_dataloader_kwargs
            if predict_dataloader_kwargs is not None
            else self._get_default_dataloader_kwargs("predict")
        )
        self.block_size = block_size
        self.collator = self.get_default_collator(  # type: ignore[arg-type]
            self.tokenizer, block_size, noise_schedule
        )
        self.prediction_collator = self.get_default_prediction_collator(  # type: ignore[arg-type]
            self.tokenizer, block_size, noise_schedule
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
            0: "prediction",
        }
        self.num_unconditional_samples = num_unconditional_samples
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.max_depth = max_depth
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.global_batch_size = global_batch_size
        self.num_dataset_workers = num_dataset_workers
        self.num_shards = 32

    def _get_cache_file(self, split: str):
        cfg_str = self.hash_cfg(self.grammar)[:8]  # shorter name
        return Path(self.manual_cache_dir) / "cfg" / cfg_str / f"{split}.txt"

    def _write_to_cache(self, cache_file: Path, sequences: list[str]):
        # create hf dataset
        try:
            logger.info(f"Writing to cache file {cache_file}")
            with open(cache_file, "w", encoding="utf-8") as f:
                for seq in sequences:
                    f.write(seq + "\n")
        except Exception as e:
            cache_file.unlink()
            raise e

    def _prepare_data_split(self):
        # Check if all cache files exist and skip if so
        cache_files = {
            split: self._get_cache_file(split)
            for split in ["train", "val", "test"]
        }
        if (
            all(cf.exists() for cf in cache_files.values())
            and not self.rewrite_manual_cache
        ):
            logger.info(
                f"Cache exists for all splits at {list(cache_files.values())[0].parent}. Skipping data generation."
            )
            return

        # Generate all possible sequences up to max_depth
        logger.info("Generating data...")
        all_sequences = list(
            map(
                lambda l_: " ".join(l_),
                generate(self.grammar, depth=self.max_depth),
            )
        )

        random.shuffle(all_sequences)
        n = len(all_sequences)
        n_train = int(self.train_size * n)
        n_val = int(self.val_size * n)
        n_test = n - n_train - n_val
        logger.info(f"n_train: {n_train}, n_val: {n_val}, n_test: {n_test}")
        train_seqs = all_sequences[:n_train]
        val_seqs = all_sequences[n_train : n_train + n_val]
        test_seqs = all_sequences[n_train + n_val :]
        # Write to cache
        for split, seqs in zip(
            ["train", "val", "test"], [train_seqs, val_seqs, test_seqs]
        ):
            cache = self._get_cache_file(split)  # type: ignore
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists() and not self.rewrite_manual_cache:
                logger.info(f"Cache file {cache} already exists. Skipping.")
                continue
            self._write_to_cache(cache, seqs)

    def prepare_data(self):
        self._prepare_data_split()

    def _get_iterable(self, split_name: str):
        # Create an iterable HuggingFace dataset for a given split
        ds = datasets.Dataset.from_text(  # type: ignore
            str(self._get_cache_file(split_name)),
            split="train",
            streaming=False,
        ).to_iterable_dataset(num_shards=self.num_shards)
        if split_name == "train" and not flags.DEBUG_OVERFIT:
            ds = ds.shuffle(buffer_size=10_000, seed=42)
        # tokenize on the fly for all splits
        ds = ds.map(
            cfg_preprocess_fn,
            batched=False,
            remove_columns=["text"],
            fn_kwargs={"tokenizer": self.tokenizer},
        )

        return ds

    def set_epoch(self, epoch: int) -> None:
        if self.train_dataset is not None:
            self.train_dataset.set_epoch(epoch)  # type: ignore

    def _setup_distributed_context(self) -> None:
        """Called after setup() is done, when dataloaders are created."""
        if self._is_ddp():
            if (trainer := self.trainer) is not None:
                if self.train_dataset is not None:
                    self.train_dataset = split_dataset_by_node(
                        self.train_dataset,
                        rank=trainer.global_rank,
                        world_size=trainer.world_size,
                    )
                if self.val_dataset is not None:
                    self.val_dataset = split_dataset_by_node(
                        self.val_dataset,
                        rank=trainer.global_rank,
                        world_size=trainer.world_size,
                    )
            else:
                raise RuntimeError("Trainer not found")

    def setup(self, stage: Optional[str] = None):
        # Setup iterable datasets per split

        if (stage == "fit" or stage is None) and self.train_dataset is None:
            self.train_dataset = self._get_iterable("train")
        if stage == "fit" or stage == "validate":
            self.val_dataset = self._get_iterable("val")
            self.predict_dataset = self.get_default_prediction_dataset(
                self.tokenizer, self.block_size, self.noise_schedule
            )
        if stage == "test":
            self.test_dataset = self._get_iterable("test")
            self.predict_dataset = self.get_default_prediction_dataset(
                self.tokenizer, self.block_size, self.noise_schedule
            )
        if stage == "predict":
            self.predict_dataset = self.get_default_prediction_dataset(
                self.tokenizer, self.block_size, self.noise_schedule
            )
        self._setup_distributed_context()
        self._check_grad_accum()

    def train_dataloader(self):
        # Use a StatefulDataLoader for deterministic iteration
        if self.train_dataset is None:
            raise ValueError("Train dataset not loaded. Call setup() first.")
        assert self.train_dataset is not None
        return StatefulDataLoader(
            self.train_dataset,  # type: ignore
            collate_fn=self.collator,
            **self.train_dataloader_kwargs,
        )

    def val_dataloader(self):
        # Eval uses standard DataLoader
        if self.val_dataset is None:
            raise ValueError("Val dataset not loaded. Call setup() first.")
        assert self.val_dataset is not None
        lm_dataloader = DataLoader(
            self.val_dataset,  # type: ignore
            collate_fn=self.collator,
            **self.val_dataloader_kwargs,
        )
        prediction_dataloader = DataLoader(
            self.predict_dataset,  # type: ignore
            collate_fn=self.prediction_collator,
            **{
                **self.val_dataloader_kwargs,
                "persistent_workers": False,
                "prefetch_factor": None,
                "num_workers": 0,
            },
        )
        return [lm_dataloader, prediction_dataloader]

    def test_dataloader(self):
        # Test returns only the LM loader
        if self.test_dataset is None:
            raise ValueError("Test dataset not loaded. Call setup() first.")
        assert self.test_dataset is not None
        lm_dataloader = DataLoader(
            self.test_dataset,  # type: ignore
            collate_fn=self.collator,
            **self.test_dataloader_kwargs,
        )
        prediction_dataloader = DataLoader(
            self.predict_dataset,  # type: ignore
            collate_fn=self.prediction_collator,
            **{
                **self.test_dataloader_kwargs,
                "persistent_workers": False,
                "prefetch_factor": None,
                "num_workers": 0,
            },
        )
        return [lm_dataloader, prediction_dataloader]

    def predict_dataloader(self):
        # Prediction uses DataLoader with no persistent or prefetch
        assert self.predict_dataset is not None
        prediction_dataloader = DataLoader(
            self.predict_dataset,  # type: ignore
            collate_fn=self.prediction_collator,
            **{
                **self.predict_dataloader_kwargs,
                "persistent_workers": False,
                "prefetch_factor": None,
                "num_workers": 0,
            },
        )
        return prediction_dataloader


# Variants of CFGDataModule for different LM tasks
class CFGDataModuleForIDLM(CFGDataModule):
    def get_tokenizer(self, vocab: List[str]):
        return IDLMSpaceTokenizer(vocab)

    def get_default_prediction_dataset(
        self,
        tokenizer,
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
            tokenizer_kwargs={"return_token_type_ids": True},
        )

    def get_default_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultIDLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultIDLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_idlm(self, batch, split, dataloader_idx)


class CFGDataModuleForILM(CFGDataModule):
    def get_tokenizer(self, vocab: List[str]):
        return ILMSpaceTokenizer(vocab)

    def get_default_prediction_dataset(
        self,
        tokenizer,
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
            tokenizer_kwargs={"return_token_type_ids": True},
        )

    def get_default_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultILMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultILMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_ilm(self, batch, split, dataloader_idx)


class CFGDataModuleForMDLM(CFGDataModule):
    def get_tokenizer(self, vocab: List[str]):
        return MDLMSpaceTokenizer(vocab)

    def get_default_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMDLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMDLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_mdlm(self, batch, split, dataloader_idx)


class CFGDataModuleForARLM(CFGDataModule):
    def get_tokenizer(self, vocab: List[str]):
        return ARLMSpaceTokenizer(vocab)

    def get_default_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultARLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: SpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultARLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_arlm(self, batch, split, dataloader_idx)
