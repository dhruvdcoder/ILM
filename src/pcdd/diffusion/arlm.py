from functools import partial
import json
from typing import (
    Any,
    Callable,
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
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from torchmetrics import (
    MeanMetric,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MetricCollection,
)
from torchmetrics.classification import BinaryAccuracy

from pcdd.datamodule.datamodule import (
    IDLMBatch,
    MDLMBatch,
    Tokenizer,
    ARLMBatch,
)
from .lightning_module_v2 import (
    BaseLightningModule,
    LossFunction,
    Predictor,
)
from pcdd.utils.rank_zero import rank_zero_only
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.nn import (
    sample_categorical,
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
)
import torch.nn.functional as F

from .idlm_v2 import Parsable

################################################################################
# region: Types


class ARLMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value. (to backprop)
        nlls (Float[TT, " num_tokens"]): The negative log likelihoods of the real predicted tokens (non-pad, and masked in input)
    """

    loss: Float[TT, ""]
    nlls: Float[TT, " num_tokens"]


class ARLMPredictionDict(TypedDict):
    """Prediction results for ARLM.

    Attributes:
        loss: None
        text: List[str]
        text_with_spl_tokens: List[str]
        ids: Integer[TT, " batch seq_len"]
        history: List[List[Tuple[str, float, int]]] the tuples are (text, time, step)
    """

    loss: None
    text: List[str]
    text_with_spl_tokens: List[str]
    ids: Integer[TT, " batch seq_len"]
    history: List[List[Tuple[str, float, int]]]


TokenLogitsType = Float[TT, " batch seq_len vocab_size"]


class ARLMStepResults(TypedDict):
    """Step results for ARLM.

    Attributes:
        x: Integer[TT, " batch seq_len"] Current predicted sequence.
        attention_mask: Bool[TT, " batch seq_len"] Mask of the current sequence.
        logits: TokenLogitsType Logits of the current sequence.
        t: Integer[TT, " batch"] Current timestep.
        change: Bool[TT, " batch"] Whether any token in the current sequence is changed.
        constraint: Bool[TT, " batch seq_len"] Constraint of the current sequence.
    """

    x: Integer[TT, " batch seq_len"]
    attention_mask: Bool[TT, " batch seq_len"]
    logits: TokenLogitsType
    change: Optional[Bool[TT, " batch"]]
    constraint: Bool[TT, " batch seq_len"]


class ARLMModel(Protocol):
    def __call__(
        self,
        x_t: Integer[TT, " batch seq_len"],
        t: Integer[TT, " batch"],
        attention_mask: Optional[Integer[TT, " batch seq_len"]] = None,
    ) -> TokenLogitsType: ...


# endregion
################################################################################

################################################################################
# region: helper functions


# endregion: helper functions
################################################################################


################################################################################
# region: Base ARLM


class ARLMLoss(LossFunction[ARLMBatch, ARLMLossDict]):
    def __init__(
        self,
        loss_on_padding: bool = False,
        model: Optional[ARLMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.loss_on_padding = loss_on_padding
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id_tensor = None

    def configure(self, pl_module: BaseLightningModule):
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: ARLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ARLMLossDict:
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: ARLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> ARLMLossDict:
        x = batch["input_ids"]
        drop = batch["drop"]
        attention_mask = batch["attention_mask"]
        model = cast(ARLMModel, self.model)
        positions = attention_mask.cumsum(dim=1) - 1
        positions *= attention_mask
        _, seq_len = attention_mask.shape
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=attention_mask.device,
            )
        )
        expanded_attention_mask = attention_mask.unsqueeze(1)
        causal_mask = expanded_attention_mask & causal_mask
        logits, _ = model(x, causal_mask, positions)
        targets = x.clone()
        targets = targets[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        attn_mask_for_loss = attention_mask[:, 1:].contiguous()
        attn_mask_for_loss = (
            attn_mask_for_loss * (1 - drop[:, 1:].contiguous())
        ).bool()
        logits_T = logits.transpose(1, 2)
        targets[~attn_mask_for_loss] = -100
        ce = torch.nn.functional.cross_entropy(
            logits_T, targets, reduction="none", ignore_index=-100
        )
        loss = (ce[attn_mask_for_loss]).mean()
        nlls = ce[attn_mask_for_loss].detach()
        return {
            "loss": loss,
            "nlls": nlls,
        }


class ARLMPredictor(torch.nn.Module, Predictor[ARLMBatch, ARLMPredictionDict]):
    token_ids_to_suppress: Integer[TT, " n_tokens_to_suppress"]

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        tokens_to_suppress: Optional[List[str]] = None,
        return_history: bool = False,
        sampling_method: Literal[
            "sample", "sample_top_k", "sample_top_p"
        ] = "sample",
        top: int = 1000,
        p: float = 0.9,
        model: Optional[ARLMModel] = None,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        # TODO (cleanup): Don't need ids to suppress.
        token_ids_to_suppress = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(token)
                for token in (
                    tokens_to_suppress
                    or [
                        tokenizer.mask_token,
                    ]
                )
            ],
            dtype=torch.long,
            requires_grad=False,
        )

        super().__init__()
        self.model = model
        self.register_buffer("token_ids_to_suppress", token_ids_to_suppress)
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        if sampling_method == "sample":
            self.sampling_function = sample_categorical
        elif sampling_method == "sample_top_k":
            self.sampling_function = partial(sample_from_top_k, top)
        elif sampling_method == "sample_top_p":
            self.sampling_function = partial(sample_from_top_p, p)
        else:
            raise ValueError(f"Invalid sampling method: {sampling_method}")

        self.return_history = return_history
        self.noise_schedule = noise_schedule

    def _predict_single_step(
        self,
        step_results: ARLMStepResults,
        current_step: int,
        final_step: bool = False,
    ) -> ARLMStepResults:
        """
        Args:
            step_results:
                x: Integer[TT, "batch seq_len"] Current predicted sequence.
                attention_mask: Bool[TT, "batch seq_len"] Mask of the current sequence.
                logits: Float[TT, "batch seq_len vocab_size"] Logits of the current sequence.
                constraint: Bool[TT, "batch seq_len"] Constraint of the current sequence.
                change: Bool[TT, "batch"] Whether any token in the current sequence is changed.
        """
        x: Integer[TT, "batch seq_len"] = step_results["x"]
        attention_mask: Bool[TT, "batch seq_len"] = step_results[
            "attention_mask"
        ]
        _, seq_len = attention_mask.shape
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=attention_mask.device,
            )
        )
        expanded_attention_mask = attention_mask.unsqueeze(1)
        causal_mask = expanded_attention_mask & causal_mask
        positions = attention_mask.cumsum(dim=1) - 1
        positions *= attention_mask
        logits, _ = self.model(x, causal_mask, positions)
        logits = logits[:, -1, :]
        if not final_step:
            probs = torch.softmax(logits, dim=-1)
            x_pred = self.sampling_function(probs)
        else:
            x_pred = torch.argmax(logits, dim=-1)
        x_pred = x_pred.unsqueeze(-1)
        eos_mask = (x == self.tokenizer.eos_token_id).any(dim=-1, keepdim=True)
        x_pred = torch.where(eos_mask, self.tokenizer.pad_token_id, x_pred)
        x = torch.cat([x, x_pred], dim=-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    device=attention_mask.device,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )
        if self.return_history:
            change = (x != step_results["x"]).any(dim=-1)  # shape (batch,)
        else:
            change = None
        return {
            "x": x,
            "attention_mask": attention_mask,
            "logits": logits,
            "constraint": None,
            "change": change,
        }

    def _stop(
        self,
        step_results: ARLMStepResults,
        current_length: int,
    ) -> bool:
        x = step_results["x"]
        max_length_reached = current_length >= self.max_length
        is_eos_reached = (x == self.tokenizer.eos_token_id).any(dim=1).all()
        return max_length_reached or is_eos_reached

    def decode(self, results: ARLMStepResults) -> Tuple[
        List[str],
        List[str],
        Integer[TT, " batch seq_len"],
    ]:
        """
        Args:
            results:
                x: Integer[TT, " batch seq_len"] Current predicted sequence.
        Returns:
            out: List[str] Decoded sequence.
            out_with_spl_tokens: List[str] Decoded sequence with special tokens.
            x: Integer[TT, " batch seq_len"] Current predicted sequence.
        """
        x: Integer[TT, " batch seq_len"] = results["x"]
        out = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            x, skip_special_tokens=False
        )
        return out, out_with_spl_tokens, x

    def _update_history(
        self,
        history: List[List[Tuple[str, float, int]]],
        step_results: ARLMStepResults,
        current_step: int,
    ) -> List[List[Tuple[str, float, int]]]:
        if not self.return_history:
            return history
        if (
            step_results["change"] is not None
            and step_results["change"].any().item()
        ):
            decoded_tuple = self.decode(step_results)
            for batch_idx in (
                step_results["change"].nonzero().flatten().tolist()
            ):
                history[batch_idx].append(
                    (
                        decoded_tuple[0][batch_idx],
                        current_step,
                    )
                )
        return history

    @torch._dynamo.disable()
    def predict(
        self,
        batch: ARLMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
        max_len: int = 0,
    ) -> ARLMPredictionDict:
        step_results: ARLMStepResults = {
            "x": batch[
                "input_ids"
            ],  # don't clone assuming that the caller is prepared for in-place operations
            "attention_mask": batch["attention_mask"],
            "logits": None,  # type: ignore ok for first step
            "constraint": batch["constraint"],
        }
        current_length = batch["input_ids"].size(1)
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_length)
        while not self._stop(step_results, current_length):
            step_results = self._predict_single_step(
                step_results, current_length
            )
            history = self._update_history(
                history, step_results, current_length
            )
            current_length += 1
        step_results = self._predict_single_step(
            step_results,
            current_length,
            final_step=True,
        )
        history = self._update_history(history, step_results, current_length)
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
        ) = self.decode(step_results)

        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "history": history,
            "loss": None,
        }

    def to_dict(
        self,
        batch: ARLMBatch,  # type: ignore
        preds: ARLMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        preds_list: List[
            Tuple[str, str, List[int], List[Tuple[str, float, int]]]
        ] = list(
            zip(
                preds["text"],
                preds["text_with_spl_tokens"],
                preds["ids"].tolist(),
                preds["history"],
            )
        )
        dicts: List[Dict[str, Any]] = []
        for text, text_with_spl_tokens, ids, history in preds_list:
            rounded_history = [
                [subseq, round(t, 4), step] for subseq, t, step in history
            ]
            dicts.append(
                {
                    "text": text,
                    "text_with_spl_tokens": text_with_spl_tokens,
                    "ids": ids,
                    "history": rounded_history,
                }
            )
        return dicts


class NLL(MeanMetric):
    pass


class Perplexity(NLL):
    def compute(self):
        return torch.exp(self.mean_value / self.weight)


class ARLMLightningModule(BaseLightningModule):
    predictor: ARLMPredictor
    loss_function: ARLMLoss

    def setup_predictor(self):
        self.predictor = hydra.utils.instantiate(
            self.config.predictor,
            tokenizer=self.tokenizer,
            noise_schedule=self.noise_schedule,
        )

    def setup_metrics(self):
        # Initialize diagnostic metrics
        nll_perplexity = MetricCollection(
            {"nll": NLL(), "perplexity": Perplexity()}
        )
        self.train_nll_perplexity = nll_perplexity.clone(prefix="train/")
        self.val_nll_perplexity = nll_perplexity.clone(prefix="val/")
        self.test_nll_perplexity = nll_perplexity.clone(prefix="test/")

    def _prepare_input_batch_for_predict(self, batch: ARLMBatch) -> ARLMBatch:
        """Use the `drop` tensor in batch to update `input_ids` and `attention_mask`.

        - `input_ids` is set to the mask token for all drop positions.
        - Tokens at drop positions are truncated.
        - The `attention_mask` is also updated to reflect the removed positions.

        Note:
            This function will do nothing if the `drop` is not set or if the `drop` tensor is all zeros.
        """
        cloned_batch: ARLMBatch = {}  # type: ignore
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                cloned_batch[k] = v.clone()
            else:
                cloned_batch[k] = v
        first_predict_position = (
            (cloned_batch["drop"] == 1).to(torch.int).argmax(dim=1)
        )
        assert (
            first_predict_position == first_predict_position[0]
        ).all(), "Var length prefix"
        first_predict_position = first_predict_position[0].item()
        if first_predict_position > 0:
            cloned_batch["input_ids"] = cloned_batch["input_ids"][
                :, :first_predict_position
            ]
            cloned_batch["attention_mask"] = cloned_batch["attention_mask"][
                :, :first_predict_position
            ]
        return cloned_batch

    def compute_loss(
        self,
        batch: ARLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Optional[Union[ARLMLossDict, ARLMPredictionDict]]:
        """
        Computes loss based on the dataloader name.

        For 'lm', the loss function is applied.
        For 'prediction', the predictor's predict_step is used.
        """
        if dataloader_name == "lm":
            return self.loss_function(
                batch, batch_idx, dataloader_idx, dataloader_name
            )
        elif dataloader_name == "prediction":
            cloned_batch = self._prepare_input_batch_for_predict(batch)
            return self.predictor.predict(
                cloned_batch,
                batch_idx,
                dataloader_idx,
                dataloader_name,
                batch["input_ids"].size(1),
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
        self.train_nll_perplexity.update(loss_dict["nlls"])

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
            self.log_dict(
                self.train_nll_perplexity,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )
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
            self.val_nll_perplexity.update(loss_dict["nlls"])
        elif dataloader_name == "prediction":
            pass  # For prediction branch, no metric update is performed for base IDLM

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
            self.log_dict(
                self.val_nll_perplexity,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )
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
            self.test_nll_perplexity.update(loss_dict["nlls"])
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
            self.log_dict(
                self.test_nll_perplexity,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )
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

    def reset_train_metrics(self) -> None:
        self.train_nll_perplexity.reset()

    def reset_val_metrics(self) -> None:
        self.val_nll_perplexity.reset()

    def reset_test_metrics(self) -> None:
        self.test_nll_perplexity.reset()


# endregion: Base ARLM
################################################################################


################################################################################
# region: ARLM for Star Graphs


#######################################################################
# region: Specialized metrics for Star Graphs


class ExactMatch(MeanMetric):
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
        matches = (
            (pred == target).all(dim=-1).to(torch.float32)
        )  # shape (batch)
        if pred_length is not None and target_length is not None:
            matches = matches * (pred_length == target_length)
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


class ARLMLightningModuleForStarGraphs(ARLMLightningModule):
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
            actual_length = (batch["input_ids"] != self.tokenizer.pad_token_id).sum(dim=-1)
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
        batch: ARLMBatch,
        preds: ARLMPredictionDict,
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

        with open(file_path, "a") as f:
            for dict_ in self.predictor.to_dict(batch, preds):
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


# endregion: ARLM for Star Graphs
################################################################################
################################################################################
# region: ARLM for Zebra

ARLMLightningModuleForZebra = ARLMLightningModuleForStarGraphs

# endregion: MDLM for Zebra
################################################################################

################################################################################
# region: ARLM for CFG


class ARLMLightningModuleForCFG(ARLMLightningModule):
    def setup_metrics(self):
        super().setup_metrics()
        self.val_parsable = Parsable(self.datamodule.parser, self.tokenizer)  # type: ignore
        self.test_parsable = Parsable(self.datamodule.parser, self.tokenizer)  # type: ignore

    def update_val_metrics(
        self, batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
    ):
        super().update_val_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            computed = self.val_parsable._just_compute(loss_dict["text"])
            loss_dict["per_sample_parsable"] = computed.bool().tolist()
            self.val_parsable.update(loss_dict["text"], computed)

    def log_val_metrics(
        self, batch, metrics, batch_idx, dataloader_idx, dataloader_name
    ):
        super().log_val_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {"val/parsable": self.val_parsable},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )

    def update_test_metrics(
        self, batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
    ):
        super().update_test_metrics(
            batch, loss_dict, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            computed = self.test_parsable._just_compute(loss_dict["text"])
            loss_dict["per_sample_parsable"] = computed.bool().tolist()
            self.test_parsable.update(loss_dict["text"], computed)

    def log_test_metrics(
        self, batch, metrics, batch_idx, dataloader_idx, dataloader_name
    ):
        super().log_test_metrics(
            batch, metrics, batch_idx, dataloader_idx, dataloader_name
        )
        if dataloader_name == "prediction":
            self.log_dict(
                {"test/parsable": self.test_parsable},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                add_dataloader_idx=False,
            )

    @rank_zero_only
    def log_predictions(
        self,
        batch: ARLMBatch,
        preds: ARLMPredictionDict,
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

        per_sample_parsable = preds.get("per_sample_parsable", None)
        fields_to_keep_in_output = list(
            set(
                ["text", "parsable"]
                + (
                    self.fields_to_keep_in_output
                    if self.fields_to_keep_in_output is not None
                    else []
                )
            )
        )

        def filter_fields(out_dict):
            if fields_to_keep_in_output is None:
                return out_dict
            return {
                k: v
                for k, v in out_dict.items()
                if k in fields_to_keep_in_output
            }

        def _to_dict(batch, preds):
            _out_dict = self.predictor.to_dict(batch, preds)
            for i, _one_out_dict in enumerate(_out_dict):
                if per_sample_parsable is not None:
                    _one_out_dict["parsable"] = bool(per_sample_parsable[i])
            return _out_dict

        text = []  # list of rows
        parsable = []  # list of rows
        with open(file_path, "a") as f:
            for dict_ in _to_dict(batch, preds):
                text.append(dict_["text"])
                parsable.append(dict_["parsable"])
                f.write(json.dumps(filter_fields(dict_)) + "\n")

        # only log one set of predictions per eval run.
        if self.trainer.global_step > getattr(
            self, "last_global_step_logged_at_which_logged_predictions", -1
        ):
            n_rows = 10
            for logger_ in logger_:
                logger_.log_text(
                    f"{split}/{dataloader_name}",
                    ["generated", "parsable"],  # column names
                    [
                        [_text, _parsable]
                        for _text, _parsable in zip(
                            text[:n_rows], parsable[:n_rows]
                        )
                    ],  # rows
                    step=self.trainer.global_step,
                )
            self.last_global_step_logged_at_which_logged_predictions = (
                self.trainer.global_step
            )


# endregion: ARLM for CFG
################################################################################
