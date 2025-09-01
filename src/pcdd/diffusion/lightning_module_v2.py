from collections.abc import Generator
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    TypedDict,
    Union,
)

import hydra
import torch
import torch.nn as nn
from torch import Tensor as TT
from lightning import LightningModule as _LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from transformers import get_scheduler
from pcdd.datamodule.datamodule import BaseDataModule, Tokenizer
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only
from .generative_perplexity import (
    GenerativePerplexityEvaluator,
    GenerativePerplexityEvaluatorResult,
)
from torchmetrics import Perplexity
import lightning as L

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


class LRSchedulerWithConfig(TypedDict):
    """We follow the same structure as the one in LightningModule.
    lr_scheduler_config = {
                # REQUIRED: The lr_scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the lr_scheduler's step size, could also be 'step'.
                # 'epoch' updates the lr_scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `lr_scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the lr_scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
            }
    """

    scheduler: LRScheduler
    interval: str  # = "step"
    frequency: int  #   = 1
    monitor: Optional[str]  # = None
    strict: bool  # = True


ModelOutput = Dict[Union[Literal["loss"], str], Any]

T_in = TypeVar("T_in", contravariant=True)
T_out = TypeVar("T_out", covariant=True)


class LossFunction(Generic[T_in, T_out], Protocol):
    model: Any
    tokenizer: Tokenizer

    def configure(self, pl_module: "BaseLightningModule"): ...

    """Called from the device context."""

    def loss_fn(
        self,
        batch: T_in,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> T_out: ...

    """Compilable part of the loss function."""

    def get_compilable_functions(self) -> List[Callable]: ...

    """Break the logic in __call__ into smaller functions that are compilable and return them here.
    """

    def __call__(
        self,
        batch: T_in,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> T_out: ...


T_out_pred = TypeVar("T_out_pred")


class Predictor(Generic[T_in, T_out_pred], Protocol):
    tokenizer: Tokenizer
    noise_schedule: NoiseSchedule
    model: Any

    def predict(
        self,
        batch: T_in,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> T_out_pred: ...

    def to_dict(
        self,
        batch: T_in,
        preds: T_out_pred,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Create json lines from the predictions batch."""
        ...


class BaseLightningModule(_LightningModule):
    """Main module that provides the scaffolding for the codebase."""

    model: nn.Module
    config: DictConfig
    predictor: Predictor
    loss_function: LossFunction
    noise_schedule: NoiseSchedule
    train_dataloader_names: Dict[int, str]
    val_dataloader_names: Dict[int, str]
    test_dataloader_names: Dict[int, str]
    predict_dataloader_names: Dict[int, str]
    predictions_dir: Path
    tokenizer: Tokenizer

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: Optional[Tokenizer] = None,
        datamodule: Optional[BaseDataModule] = None,
        fields_to_keep_in_output: Optional[List[str]] = None,
        write_per_sample_metrics: bool = False,
        **kwargs: Any,
    ):
        # Ensure config is a DictConfig. When loading from checkpoint, we get a dict instance.
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        # If an override config is provided, merge it.
        if "cfg" in kwargs:
            override_cfg = kwargs.pop("cfg")
            if not isinstance(override_cfg, DictConfig):
                override_cfg = OmegaConf.create(override_cfg)
            _cfg = OmegaConf.merge(cfg, override_cfg)
            assert isinstance(_cfg, DictConfig)
            cfg = _cfg
        super().__init__()
        self.config = cfg
        self.datamodule = datamodule  # hold a reference to the datamodule
        self.update_omegaconf_resolvers()
        self.setup_vocab(tokenizer)
        self.instantiate_noise_schedule()
        self.instantiate_model()
        self.instantiate_predictor()
        self.instantiate_loss_function()
        self.setup_dataloader_names()
        self.setup_metrics()
        self.predictions_dir = Path(cfg.paths.run_dir) / "predictions"
        if (
            cfg.lightning_module.get("fields_to_keep_in_output", None)
            is not None
        ):
            fields_to_keep_in_output = (
                cfg.lightning_module.fields_to_keep_in_output
            )
        self.fields_to_keep_in_output = fields_to_keep_in_output
        # validate and save config at the end
        self.validate_config(self.config)
        # Save hyperparameters as a plain dict.
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        # Dictionary to store whether we have printed the batch
        self.printed_batches: Set[str] = set()
        self.last_global_step_logged_at_which_logged_predictions: int = -1
        if cfg.lightning_module.get("write_per_sample_metrics", False):
            self.write_per_sample_metrics = True
        else:
            self.write_per_sample_metrics = write_per_sample_metrics

    ############################################################
    # region: Setup methods

    def setup_vocab(self, tokenizer: Optional[Tokenizer]):
        if tokenizer is None:
            self.tokenizer = hydra.utils.instantiate(
                self.config.datamodule.tokenizer
            )
        else:
            self.tokenizer = tokenizer

    def update_omegaconf_resolvers(self):
        OmegaConf.clear_resolver("tokenizer")
        OmegaConf.register_new_resolver(
            "tokenizer", lambda x: getattr(self.tokenizer, x)
        )
        OmegaConf.clear_resolver("datamodule")
        OmegaConf.register_new_resolver(
            "datamodule", lambda x: getattr(self.datamodule, x)
        )
        OmegaConf.clear_resolver("lightning_module")
        OmegaConf.register_new_resolver(
            "lightning_module", lambda x: getattr(self, x)
        )

    def instantiate_noise_schedule(self):
        # first try to get it from the datamodule
        if self.datamodule is not None:
            if hasattr(self.datamodule, "noise_schedule"):
                self.noise_schedule = self.datamodule.noise_schedule  # type: ignore
                return
        # otherwise, instantiate it from the config
        self.noise_schedule = hydra.utils.instantiate(
            self.config.noise_schedule
        )

    def instantiate_model(self):
        self.model = hydra.utils.instantiate(self.config.model)

    def instantiate_predictor(self):
        self.predictor = hydra.utils.instantiate(self.config.predictor)
        if self.predictor.tokenizer is None:
            self.predictor.tokenizer = self.tokenizer
        if self.predictor.noise_schedule is None:
            self.predictor.noise_schedule = self.noise_schedule
        if self.predictor.model is None:
            self.predictor.model = self.model

    def instantiate_loss_function(self):
        self.loss_function = hydra.utils.instantiate(self.config.loss)
        if self.loss_function.tokenizer is None:
            self.loss_function.tokenizer = self.tokenizer
        if self.loss_function.model is None:
            self.loss_function.model = self.model
        # check for consistency with the predictor
        if hasattr(self, "check_loss_predictor_consistency"):
            self.check_loss_predictor_consistency()

    def setup_dataloader_names(self):
        if self.datamodule is None:
            self.train_dataloader_names = {}
            self.val_dataloader_names = {}
            self.test_dataloader_names = {}
            self.predict_dataloader_names = {}
        else:
            self.train_dataloader_names = (
                self.datamodule.train_dataloader_names
            )
            self.val_dataloader_names = self.datamodule.val_dataloader_names
            self.test_dataloader_names = self.datamodule.test_dataloader_names
            self.predict_dataloader_names = (
                self.datamodule.predict_dataloader_names
            )

    def setup_metrics(self):
        raise NotImplementedError

    def validate_config(self, cfg: DictConfig):
        return

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        # Lightning Hook called when the LightningModule is initialized on each device.
        if stage in ["fit", "validate"]:
            try:
                self.train_dataloader_names = (
                    self.trainer.datamodule.train_dataloader_names
                )
                self.val_dataloader_names = (
                    self.trainer.datamodule.val_dataloader_names
                )
                self.test_dataloader_names = (
                    self.trainer.datamodule.test_dataloader_names
                )
                self.predict_dataloader_names = (
                    self.trainer.datamodule.predict_dataloader_names
                )
            except AttributeError:
                logger.warning(
                    "Could not set dataloader names for datamodule. Either you are not using a datamodule,"
                    " or the datamodule does not support setting the dataloader "
                    "names using `<train,val,test,predict>_dataloader_names`."
                )

    def configure_model(self):
        if self.loss_function is not None:
            if hasattr(self.loss_function, "configure"):
                self.loss_function.configure(self)
        if self.config.get("compile", False):
            # We need to execute the prediction loop in eager mode, but torch.compile.set_stance
            # is only available in torch >= 2.6. In torch 2.5, there is no way to tell torch to
            # execute a compiled model in eager mode. So we will wait for lightning to support torch 2.6.
            # logger.info("Wrapping model in torch.compile")
            # self.model.compile(
            #    fullgraph=True,
            #    options={
            #        "trace.graph_diagram": False,
            #        "trace.enabled": False,
            #    },
            # )  # type: ignore

            # In the meantime, we will compile the loss function callable
            torch._dynamo.config.cache_size_limit = 100  # type: ignore
            logger.info("Compiling loss function")
            self.loss_function.loss_fn = torch.compile(
                self.loss_function.loss_fn,
                fullgraph=True,
                options={
                    "trace.graph_diagram": False,
                    "trace.enabled": False,
                },
            )  # type: ignore

    @staticmethod
    def create_lr_scheduler(
        optimizer: Optimizer,
        name: str,
        num_warmup_steps: Optional[int] = None,
        fraction_warmup_steps: Optional[float] = None,
        num_training_steps: Optional[int] = None,
        interval: Literal["step", "epoch"] = "step",
        frequency: int = 1,
        monitor: Optional[str] = "train_loss",
        strict: bool = True,
        **kwargs: Any,
    ) -> LRSchedulerWithConfig:
        """Creates a learning rate noise_schedule with the given configuration.

        Args:
            name: Huggingface name of the learning rate noise_schedule. https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.get_scheduler
            optimizer: The optimizer to use with the noise_schedule
            num_training_steps: The total number of training steps.
            num_warmup_steps: The number of warmup steps.
            fraction_warmup_steps: The fraction of training steps to use for warmup.
            interval: The interval at which to update the learning rate.
            frequency: The frequency of the learning rate updates.
            monitor: The metric to monitor for the learning rate noise_schedule.
            strict: Whether to strictly follow the learning rate schedule.
            **kwargs: Additional keyword arguments to pass to the learning rate noise_schedule.
        """
        if num_warmup_steps is None:
            if fraction_warmup_steps is None:
                raise ValueError(
                    "Either num_warmup_steps or fraction_warmup_steps must be provided"
                )
            if num_training_steps is None:
                raise ValueError(
                    "num_training_steps must be provided when using fraction_warmup_steps"
                )
            num_warmup_steps = int(num_training_steps * fraction_warmup_steps)
        logger.info(f"num_warmup_steps: {num_warmup_steps}")

        lr_scheduler = get_scheduler(
            name, optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
        return LRSchedulerWithConfig(
            scheduler=lr_scheduler,
            interval=interval,
            frequency=frequency,
            monitor=monitor,
            strict=strict,
        )

    def configure_optimizers(
        self,
    ) -> Dict[
        Literal["optimizer", "lr_scheduler"],
        Union[Optimizer, LRSchedulerWithConfig],
    ]:
        partial_optimizer = hydra.utils.instantiate(
            self.config.optimizer, _partial_=True
        )
        main_params_with_weight_decay = list(
            p for _, p in self.model.get_named_params_for_weight_decay()
        )
        main_params_without_weight_decay = list(
            p for _, p in self.model.get_named_params_for_no_weight_decay()
        )
        logger.info(
            f"Num params with weight decay in the `model`: {len(main_params_with_weight_decay)}"
        )
        logger.info(
            f"Num params without weight decay in the `model`: {len(main_params_without_weight_decay)}"
        )
        groups = [
            {"params": main_params_with_weight_decay},
            {"params": main_params_without_weight_decay, "weight_decay": 0.0},
        ]
        optimizer = partial_optimizer(groups)
        lr_scheduler: LRSchedulerWithConfig = self.create_lr_scheduler(
            optimizer, **self.config.lr_scheduler
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    # endregion: Setup methods
    ############################################################

    ############################################################
    # region: Task-specific methods

    def compute_loss(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def update_train_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def update_val_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def update_test_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def update_predict_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def log_train_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def reset_train_metrics(self):
        raise NotImplementedError

    def log_val_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def reset_val_metrics(self):
        raise NotImplementedError

    def log_test_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def reset_test_metrics(self):
        raise NotImplementedError

    def log_predict_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def reset_predict_metrics(self):
        raise NotImplementedError

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        dataloader_name: Optional[str] = self.train_dataloader_names.get(
            dataloader_idx if dataloader_idx else -1, "lm"
        )
        loss_dict = self.compute_loss(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        if "loss" in loss_dict:
            if bool(loss_dict["loss"].isnan()):
                global_step = self.trainer.global_step
                raise RuntimeError(
                    f"NaN loss encountered in training step {global_step} in epoch {self.trainer.current_epoch}"
                )
            if bool(loss_dict["loss"].isinf()):
                global_step = self.trainer.global_step
                raise RuntimeError(
                    f"Inf loss ({loss_dict['loss']}) encountered in training step {global_step} in epoch {self.trainer.current_epoch}"
                )

        self.update_train_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        self.log_train_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    @torch._dynamo.disable()
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        dataloader_name: Optional[str] = self.val_dataloader_names.get(
            dataloader_idx if dataloader_idx else -1, "lm"
        )
        loss_dict = self.compute_loss(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        self.update_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        self.log_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    @torch._dynamo.disable()
    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        dataloader_name: Optional[str] = self.test_dataloader_names.get(
            dataloader_idx if dataloader_idx else -1, "lm"
        )
        loss_dict = self.compute_loss(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        self.update_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        self.log_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    @torch._dynamo.disable()
    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        dataloader_name: Optional[str] = self.predict_dataloader_names.get(
            dataloader_idx if dataloader_idx else -1, "lm"
        )
        loss_dict = self.predictor.predict(
            batch, batch_idx, dataloader_idx, dataloader_name
        )

        self.update_predict_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    # endregion: Task-specific methods
    ############################################################

    ############################################################
    # region: Lightning Hooks

    def on_train_batch_start(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        if "train/0" in self.printed_batches:
            return
        self.trainer.datamodule.print_batch(batch, "train", 0)
        self.printed_batches.add("train/0")

    def on_validation_batch_start(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if f"val/{dataloader_idx}" in self.printed_batches:
            return
        self.trainer.datamodule.print_batch(batch, "val", dataloader_idx)
        self.printed_batches.add(f"val/{dataloader_idx}")

    def on_test_batch_start(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if f"test/{dataloader_idx}" in self.printed_batches:
            return
        self.trainer.datamodule.print_batch(batch, "test", dataloader_idx)
        self.printed_batches.add(f"test/{dataloader_idx}")

    def on_train_epoch_start(self) -> None:
        self.model.train()
        try:
            self.trainer.datamodule.set_epoch(self.trainer.current_epoch)
        except AttributeError:
            logger.warning(
                "Could not set epoch for datamodule. Either you are not using a datamodule,"
                " or the datamodule does not support setting the epoch using `set_epoch()`."
                "This could lead to no shuffling in the dataloader between epoch if using IterableDataset."
            )

    def on_train_epoch_end(self) -> None:
        # reset train metrics if not passing metric objects to the log()
        pass

    def on_validation_epoch_start(self) -> None:
        self.model.eval()

    def on_validation_epoch_end(self) -> None:
        # reset val metrics if not passing metric objects to the log()
        pass

    def on_test_epoch_start(self) -> None:
        self.model.eval()

    def on_test_epoch_end(self) -> None:
        # reset test metrics if not passing metric objects to the log()
        pass

    # endregion: Lightning Hooks
    ############################################################

    ############################################################
    # region: other utilities

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # CLEANUP: This method is to load old models that serialize rotary embedding buffers
        # Get the state_dict from the checkpoint (the key might be "state_dict")
        state_dict = checkpoint.get(
            "state_dict", checkpoint
        )  # entier lightning module's state_dict
        # Identify the keys corresponding to the rotary buffers. They will be in .model as well as .predictor.model
        ranked_logger.info(
            "Removing rotary embedding buffers from the state_dict."
        )
        keys_to_remove = [
            key
            for key in state_dict
            if "rotary_emb.sin" in key or "rotary_emb.cos" in key
        ]
        for key in keys_to_remove:
            state_dict.pop(key)

    def top_level_named_modules(
        self,
    ) -> Generator[Tuple[str, nn.Module], None, None]:
        yield "model", self.model

    def get_predictions_file_path(
        self,
        split: Literal["train", "val", "test"],
        dataloader_name: str,
        epoch: int,
        step: int,
    ) -> Path:
        return (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )

    def get_unconditional_samples_file_path(
        self,
        trainer: L.Trainer,
    ) -> Path:
        return self.get_predictions_file_path(
            "val", "prediction", trainer.current_epoch, trainer.global_step
        )

    @rank_zero_only
    def log_predictions(
        self,
        batch: Dict[str, Any],
        preds: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_name: str,
    ):
        step = self.trainer.global_step or 0
        epoch = self.trainer.current_epoch or 0
        file_path = self.get_predictions_file_path(
            split, dataloader_name, epoch, step
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger_ = [
            l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")
        ]
        text = []  # list of rows

        with open(file_path, "a") as f:
            for dict_ in self.predictor.to_dict(
                batch, preds, dataloader_name=dataloader_name
            ):
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
                    ["text"],  # column names
                    [[_text] for _text in text[:n_rows]],  # rows
                    step=self.trainer.global_step,
                )
            self.last_global_step_logged_at_which_logged_predictions = (
                self.trainer.global_step
            )

    # endregion: other utilities
    ############################################################


############################################################
# region: Callbacks


class GenerativePerplexityCallback(L.Callback):
    def __init__(
        self,
        evaluator: GenerativePerplexityEvaluator,
    ):
        """
        Args:
            evaluator: Any generic evaluator that implements the protocol.
        """
        self.evaluator = evaluator
        ignore_index = evaluator.ignore_index
        assert ignore_index is not None
        logger.info(
            f"Using ignore_index: {ignore_index} for generative perplexity with "
            f"Evaluator: {evaluator}"
        )

    @torch.inference_mode()
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: BaseLightningModule
    ) -> None:
        # check for the samples on the disk
        model_was_training = pl_module.training
        # pl_module.model.eval()
        file_path = pl_module.get_unconditional_samples_file_path(trainer)

        # if model_was_training:
        #    pl_module.model.train()

        perplexity = self.compute_generative_perplexity(
            self.evaluator, pl_module, file_path
        )
        if perplexity is not None:

            trainer._logger_connector.log_metrics(
                {
                    f"val/generative_perplexity/{self.evaluator.name}": perplexity
                },
            )

    @staticmethod
    def compute_generative_perplexity(
        evaluator: GenerativePerplexityEvaluator,
        pl_module: BaseLightningModule,
        file_path: Path,
    ) -> Optional[TT]:
        if not file_path.exists():
            logger.error(
                f"No samples found at {file_path}. "
                "If you are using a callback like UnconditionalSampleGenerator, to generate samples, "
                "make sure that it is placed before GenerativePerplexity in the callbacks list."
            )
            return None

        # Load samples from file and evaluate in batches
        samples = []
        generative_perplexity_metric = Perplexity(
            ignore_index=evaluator.ignore_index
        ).to(pl_module.device)
        generative_perplexity_metric.reset()
        with evaluator.loaded(
            pl_module.tokenizer,  # type: ignore
            pl_module.device,
        ):  # load it on the same device for now
            with open(file_path) as f:
                for line in f:
                    sample = json.loads(line)
                    text = sample["text"]
                    samples.append(text)

                    if len(samples) == evaluator.batch_size:
                        result: Optional[
                            GenerativePerplexityEvaluatorResult
                        ] = evaluator(samples)
                        # reset the samples list for the next batch
                        samples = []
                        if result is None:
                            continue
                        logits = result["logits"]
                        target = result["target"]
                        generative_perplexity_metric.update(logits, target)

                # Handle remaining samples in last batch if any
                if samples:
                    result = evaluator(samples)
                    if result is not None:
                        logits = result["logits"]
                        target = result["target"]
                        generative_perplexity_metric.update(logits, target)

        perplexity = generative_perplexity_metric.compute()
        return perplexity


# endregion: Callbacks
############################################################
