import json
import time
from functools import partial
from itertools import cycle
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import hydra
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from torchmetrics import (
    MeanAbsolutePercentageError,
    MeanMetric,
    MeanSquaredError,
    MetricCollection,
)

from pcdd import flags
from pcdd.datamodule.datamodule import ARLMBatch, ITBatch, Tokenizer
from transformers import GenerationConfig
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
)
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only

from .lightning_module_v2 import BaseLightningModule, Predictor, LossFunction
from .nn import general_sample_over_last_two_dims, remove_tokens
from pcdd.datamodule.xlnet.datamodule_xlnet import (
    XLNetTrainingBatch,
    XLNetBaseBatch,
    XLNetPredictionBatch,
)
from pcdd.models.xlnet.modeling_xlnet import XLNetLMHeadModelOutput

logger = RankedLogger(__name__, rank_zero_only=True)

################################################################################
# region: Types


class XLNetLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value.
    """

    loss: Float[TT, ""]
    logits: Float[TT, "batch_seq_len vocab_size"]


class XLNetPredictionDict(TypedDict):
    """Output of the Predictor for IT.

    Attributes:
        loss (Optional[Float[TT, "batch"]]): The loss value. Typically None.
        text (List[str]): The batch of generated text without special tokens.
        text_with_spl_tokens (List[str]): The batch of generated text with special tokens.
        ids (Integer[TT, " batch seq_len"]): The batch of generated token_ids.
        attention_mask (Bool[TT, " batch seq_len"]): Attention mask accompanying the generated ids.
        positions (Integer[TT, " batch seq_len"]): The batch of positions of the generated tokens accompanying the ids.
        history (List[List[Tuple[str, float, int]]]): The batch of history.
            Each entry is a list of tuples, where each tuple contains
            (current_string, time, step_number) of when some change is made to the generated string.
    """

    text: List[str]
    text_with_spl_tokens: List[str]
    ids: Integer[TT, " batch seq_len"]
    time_taken: List[float]


TokenLogitsType = Float[TT, " batch seq_len vocab_size"]


class XLNetModel(Protocol):
    def __call__(
        self,
        input_ids: Integer[TT, " batch seq_len"],
        attention_mask: Integer[TT, " batch seq_len"],
        perm_mask: Integer[TT, " batch seq_len seq_len"],
        target_mapping: Optional[
            Integer[TT, " batch num_predict seq_len"]
        ] = None,
        token_type_ids: Optional[Integer[TT, " batch seq_len"]] = None,
        labels: Optional[Integer[TT, " batch num_predict"]] = None,
    ) -> Tuple[Float[TT, "batch"], Float[TT, "batch_seq_len vocab_size"]]: ...


# endregion
################################################################################


################################################################################
# region: Base XLNet


################################################################
# region: Loss functions
# We use the loss function built into the HuggingFace XLNet model.
class XLNetLoss(LossFunction[XLNetTrainingBatch, XLNetLossDict]):
    def __init__(
        self,
        model: Optional[XLNetModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer  # type: ignore

    def configure(self, pl_module: BaseLightningModule):
        pass

    def __call__(
        self,
        batch: XLNetTrainingBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> XLNetLossDict:
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: XLNetTrainingBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> XLNetLossDict:
        model = cast(XLNetModel, self.model)
        loss, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            perm_mask=batch["perm_mask"],
            target_mapping=batch["target_mapping"],
            labels=batch["labels"],
        )

        return {
            "loss": loss,
            "logits": logits.detach(),
        }


# endregion: Loss functions
################################################################

###############################################################
# region: Predictors


class XLNetPredictorUtilitiesMixin:
    tokenizer: Tokenizer

    def to_dict(
        self,
        batch: XLNetPredictionBatch,
        preds: XLNetPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        preds_list: List[Tuple[str, str, List[int], float]] = list(
            zip(
                preds["text"],
                preds["text_with_spl_tokens"],
                preds["ids"].tolist(),
                preds.get(
                    "time_taken", cycle([-1])
                ),  # -1 when the predict method does not measure time.
            )
        )
        dicts: List[Dict[str, Any]] = []
        for text, text_with_spl_tokens, ids, time_taken in preds_list:

            dicts.append(
                {
                    "text": text,
                    "text_with_spl_tokens": text_with_spl_tokens,
                    "ids": ids,
                    "time_taken": time_taken,
                    "history": [],
                }
            )
        return dicts


class HFXLNetPredictor(
    torch.nn.Module,
    XLNetPredictorUtilitiesMixin,
    Predictor[Any, XLNetPredictionDict],
):
    """For now we are going to use hf_model.generate(inputs, generation_config) to decode in left-to-right manner."""

    def __init__(
        self,
        max_length: int,
        top: int = 10,
        p: float = 0.9,
        sampling_method: Literal[
            "sample", "sample_top_k", "sample_top_p"
        ] = "sample_top_k",
        model: Optional[XLNetModel] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        tokenizer: Optional[Tokenizer] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        if generation_config is None:
            self.generation_config = GenerationConfig(
                do_sample=(
                    not (sampling_method == "sample_from_top_k" and top == 1)
                ),
                num_beams=1,
                max_length=max_length,
                return_dict_in_generate=False,
            )
        else:
            self.generation_config = generation_config
        self.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if sampling_method == "sample_top_k":
            self.generation_config.top_k = top
        elif sampling_method == "sample_top_p":
            self.generation_config.top_p = p
        else:
            assert sampling_method == "sample"

    @torch._dynamo.disable()
    def predict(
        self,
        batch: XLNetPredictionBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> XLNetPredictionDict:
        _start_time = time.time()
        hf_outputs = self.model._model.generate(
            input_ids=batch["input_ids"],
            generation_config=self.generation_config,
            attention_mask=batch["attention_mask"],
        )
        _end_time = time.time()
        _time_taken = _end_time - _start_time
        out = self.tokenizer.batch_decode(hf_outputs, skip_special_tokens=True)
        out_with_spl_tokens = self.tokenizer.batch_decode(
            hf_outputs, skip_special_tokens=False
        )
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": hf_outputs,
            "time_taken": [_time_taken] * len(out),
        }


# endregion: Predictors
###############################################################


class NLL(MeanMetric):
    pass


class Perplexity(NLL):
    def compute(self):
        return torch.exp(self.mean_value / self.weight)


class XLNetLightningModule(BaseLightningModule):
    predictor: HFXLNetPredictor

    def setup_predictor(self):
        self.predictor = hydra.utils.instantiate(
            self.config.predictor,
            tokenizer=self.tokenizer,
            # noise_schedule=self.noise_schedule,
        )

    def setup_metrics(self):
        # Initialize diagnostic metrics
        nll_perplexity = MetricCollection(
            {"nll": NLL(), "perplexity": Perplexity()}
        )
        self.train_nll_perplexity = nll_perplexity.clone(prefix="train/")
        self.val_nll_perplexity = nll_perplexity.clone(prefix="val/")
        self.test_nll_perplexity = nll_perplexity.clone(prefix="test/")

    def _prepare_input_batch_for_predict(
        self, batch: XLNetPredictionBatch
    ) -> XLNetPredictionBatch:
        """Use the `drop` tensor in batch to update `input_ids` and `attention_mask`.

        - `input_ids` is set to the mask token for all drop positions.
        - Tokens at drop positions are truncated.
        - The `attention_mask` is also updated to reflect the removed positions.

        Note:
            This function will do nothing if the `drop` is not set or if the `drop` tensor is all zeros.
        """
        cloned_batch: XLNetBaseBatch = {}  # type: ignore
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                cloned_batch[k] = v.clone()
            else:
                cloned_batch[k] = v
        first_predict_position = batch["drop"].int().argmax(dim=-1)
        assert (
            first_predict_position == first_predict_position[0]
        ).all(), "Var length prefix not supported. Use left padding"
        first_predict_position = first_predict_position[0].item()
        if first_predict_position > 0:
            cloned_batch["input_ids"] = cloned_batch["input_ids"][
                :, :first_predict_position
            ]
            cloned_batch["attention_mask"] = cloned_batch["attention_mask"][
                :, :first_predict_position
            ]
            cloned_batch["drop"] = cloned_batch["drop"][
                :, :first_predict_position
            ]
        return cloned_batch

    def compute_loss(
        self,
        batch: Union[XLNetTrainingBatch, XLNetPredictionBatch],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Optional[Union[XLNetLossDict, XLNetPredictionDict]]:
        """
        Computes loss based on the dataloader name.

        For 'lm', the loss function is applied.
        For 'prediction', the predictor's predict_step is used.
        """
        if dataloader_name == "lm":
            return self.loss_function(
                cast(XLNetTrainingBatch, batch),
                batch_idx,
                dataloader_idx,
                dataloader_name,
            )
        elif dataloader_name == "prediction":
            cloned_batch = self._prepare_input_batch_for_predict(
                cast(XLNetPredictionBatch, batch)
            )
            return self.predictor.predict(
                cloned_batch,
                batch_idx,
                dataloader_idx,
                dataloader_name,
            )
        else:  # let derived class call with a different dataloader name
            return None

    def update_train_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update training metrics with values computed by the loss function.
        """
        pass
        # self.train_nll_perplexity.update(loss_dict["nlls"])

    def log_train_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log training metrics.
        """
        if dataloader_name == "lm":
            self.log(
                "train/loss",
                metrics["loss"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=True,
                logger=True,
                add_dataloader_idx=False,
            )
            # self.log_dict(
            #    self.train_nll_perplexity,
            #    on_step=False,
            #    on_epoch=True,
            #    prog_bar=False,
            #    add_dataloader_idx=False,
            # )
            # Optionally, remove non-essential elements from metrics
            for key in [
                "nlls",
            ]:
                metrics.pop(key, None)

    def update_val_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update validation metrics.
        """
        if dataloader_name == "lm":
            # self.val_nll_perplexity.update(loss_dict["nlls"])
        elif dataloader_name == "prediction":
            pass  # For prediction branch, no metric update is performed for base model

    def log_val_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log validation metrics.
        """
        if dataloader_name == "lm":
            # self.log_dict(
            #    self.val_nll_perplexity,
            #    on_step=False,
            #    on_epoch=True,
            #    prog_bar=False,
            #    add_dataloader_idx=False,
            # )
            for key in [
                "nlls",
            ]:
                metrics.pop(key, None)
        elif dataloader_name == "prediction":
            # Write to file and logger.
            self.log_predictions(batch, metrics, "val", dataloader_name)

    def update_test_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update test metrics.
        """
        if dataloader_name == "lm":
            pass
            # self.test_nll_perplexity.update(loss_dict["nlls"])
        elif dataloader_name == "prediction":
            pass  # For prediction branch, no metric update is performed for base

    def log_test_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log test metrics.
        """
        if dataloader_name == "lm":
            # self.log_dict(
            #    self.test_nll_perplexity,
            #    on_step=False,
            #    on_epoch=True,
            #    prog_bar=False,
            #    add_dataloader_idx=False,
            # )
            for key in [
                "nlls",
            ]:
                metrics.pop(key, None)
        elif dataloader_name == "prediction":
            # Write to file and logger.
            self.log_predictions(batch, metrics, "test", dataloader_name)

    def update_predict_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Update metrics during prediction (if needed).
        Currently, no extra metrics are tracked for prediction.
        """
        pass

    def log_predict_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        """
        Log prediction metrics (if needed).
        """
        pass


# endregion: Base XLNet
################################################################################


################################################################################
# region: XLNet for Star Graphs


#######################################################################
# region: Specialized metrics for Star Graphs


class ExactMatch(MeanMetric):
    def _just_compute(
        self,
        pred: Integer[TT, " batch seq_len"],
        target: Integer[TT, " batch seq_len"],
        pred_length: Optional[Integer[TT, " batch"]] = None,
        target_length: Optional[Integer[TT, " batch"]] = None,
    ) -> Float[TT, " batch"]:
        """
        Args:
            pred: predicted tokens
            target: target tokens
            pred_length: length of the predicted tokens
            target_length: length of the target tokens
        """
        matches = (
            (pred == target).all(dim=-1).to(torch.float32)
        )  # shape (batch)
        if pred_length is not None and target_length is not None:
            matches = matches * (pred_length == target_length)
        return matches

    def update(
        self,
        pred: Integer[TT, " batch seq_len"],
        target: Integer[TT, " batch seq_len"],
        pred_length: Optional[Integer[TT, " batch"]] = None,
        target_length: Optional[Integer[TT, " batch"]] = None,
    ):
        """
        Args:
            pred: predicted tokens
            target: target tokens
            pred_length: length of the predicted tokens
            target_length: length of the target tokens
        """
        matches = self._just_compute(pred, target, pred_length, target_length)
        super().update(matches)


class HammingAccuracy(MeanMetric):
    def update(
        self,
        pred: Integer[TT, " batch seq_len"],
        target: Integer[TT, " batch seq_len"],
        pred_mask: Optional[Integer[TT, " batch seq_len"]] = None,
    ):
        """
        Args:
            pred: predicted tokens
            target: target tokens
            pred_mask: True for positions that predicted.
        """
        if pred_mask is None:
            pred_mask = torch.ones_like(pred, dtype=torch.bool)
        temp = (pred == target) * pred_mask
        correct = temp.sum(dim=-1).to(torch.float32)  # shape (batch)
        total = pred_mask.sum(dim=-1).to(torch.float32)  # shape (batch)
        acc = correct / total  # shape (batch)
        super().update(acc)


# endregion: Specialized metrics for Star Graphs
#######################################################################


class XLNetLightningModuleForStarGraphs(XLNetLightningModule):
    def _get_generated_length(
        self, pred_ids: Integer[TT, " batch seq_len"]
    ) -> Integer[TT, " batch"]:
        pad_token_id = int(cast(int, self.tokenizer.pad_token_id))
        remove = torch.tensor([pad_token_id], device=pred_ids.device)
        return torch.isin(pred_ids, remove, invert=True).sum(dim=-1)

    def _pad_truncate(self, x, max_len):
        batch, seq_len = x.shape
        if seq_len < max_len:
            padding = (0, max_len - seq_len)
            x = F.pad(x, padding, value=self.tokenizer.pad_token_id)
        else:
            x = x[:, :max_len]
        return x

    def setup_metrics(self):
        super().setup_metrics()
        self.val_generated_length_rmse = MeanSquaredError(squared=False)
        self.test_generated_length_rmse = MeanSquaredError(squared=False)
        self.val_generated_length_mape = MeanAbsolutePercentageError()
        self.test_generated_length_mape = MeanAbsolutePercentageError()
        self.val_exact_match = ExactMatch()
        self.test_exact_match = ExactMatch()
        self.val_hamming_accuracy = HammingAccuracy()
        self.test_hamming_accuracy = HammingAccuracy()

    def update_val_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            generated_length = self._get_generated_length(loss_dict["ids"])
            actual_length = self._get_generated_length(batch["input_ids"])

            self.val_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.val_generated_length_mape.update(
                generated_length, actual_length
            )
            loss_dict["ids"] = self._pad_truncate(
                loss_dict["ids"], batch["input_ids"].size(1)
            )
            self.val_exact_match.update(
                loss_dict["ids"],
                batch["input_ids"],
                pred_length=generated_length,
                target_length=actual_length,
            )
            self.val_hamming_accuracy.update(
                loss_dict["ids"], batch["input_ids"], pred_mask=batch["drop"]
            )
        elif dataloader_name == "lm":
            pass  # parent class handles this
        else:
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_val_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_val_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "val/generated_length_rmse": self.val_generated_length_rmse,
                    "val/generated_length_mape": self.val_generated_length_mape,
                    "val/exact_match": self.val_exact_match,
                    "val/hamming_accuracy": self.val_hamming_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def update_test_metrics(
        self,
        batch: Dict[str, Any],
        loss_dict: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().update_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            generated_length = self._get_generated_length(loss_dict["ids"])
            actual_length = batch["attention_mask"].sum(dim=-1)
            self.test_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.test_generated_length_mape.update(
                generated_length, actual_length
            )
            loss_dict["ids"] = self._pad_truncate(
                loss_dict["ids"], batch["input_ids"].size(1)
            )
            self.test_exact_match.update(
                loss_dict["ids"],
                batch["input_ids"],
                pred_length=generated_length,
                target_length=actual_length,
            )
            self.test_hamming_accuracy.update(
                loss_dict["ids"], batch["input_ids"], pred_mask=batch["drop"]
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    def log_test_metrics(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> None:
        super().log_test_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {
                    "test/generated_length_rmse": self.test_generated_length_rmse,
                    "test/generated_length_mape": self.test_generated_length_mape,
                    "test/exact_match": self.test_exact_match,
                    "test/hamming_accuracy": self.test_hamming_accuracy,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        elif dataloader_name != "lm":
            raise ValueError(f"Unknown dataloader name: {dataloader_name}")

    @rank_zero_only
    def log_predictions(
        self,
        batch: XLNetPredictionBatch,
        preds: XLNetPredictionDict,
        split: Literal["train", "val", "test"],
        dataloader_name: str,
    ):
        step = self.trainer.global_step or 0
        epoch = self.trainer.current_epoch or 0
        file_path = (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger_ = [
            l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")
        ]
        text = []  # list of rows
        ground_truth_text = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        # Compute per-sample exact match
        pred_ids = preds["ids"]
        target_ids = batch["input_ids"]
        generated_length = self._get_generated_length(pred_ids)
        actual_length = self._get_generated_length(target_ids)
        per_sample_exact_match = self.val_exact_match._just_compute(
            pred_ids,
            target_ids,
            pred_length=generated_length,
            target_length=actual_length,
        )

        def _to_dict(batch, preds):
            _out_dict = self.predictor.to_dict(batch, preds)
            for i, _one_out_dict in enumerate(_out_dict):
                _one_out_dict["truth"] = ground_truth_text[i]
                _one_out_dict["exact_match"] = bool(
                    per_sample_exact_match[i].item()
                )
            return _out_dict

        with open(file_path, "a") as f:
            for dict_ in _to_dict(batch, preds):
                text.append(dict_["text"])
                f.write(json.dumps(dict_) + "\n")
        n_rows = 10
        for logger_ in logger_:
            logger_.log_text(
                f"{split}/{dataloader_name}",
                ["generated", "truth"],  # column names
                [
                    [_text, _gt]
                    for _text, _gt in zip(
                        text[:n_rows], ground_truth_text[:n_rows]
                    )
                ],  # rows
                step=self.trainer.global_step,
            )


# endregion: XLNet for Star Graphs
################################################################################
