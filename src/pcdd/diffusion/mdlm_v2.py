from functools import partial
import json
import time
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
from itertools import cycle
import torch.nn.functional as F
import torch.distributions as dists

################################################################################
# region: Types


class MDLMLossDict(TypedDict):
    """Output of the LossFunction Callable.

    Attributes:
        loss (Float[TT, ""]): The total loss value. (to backprop)
        nlls (Float[TT, " num_tokens"]): The negative log likelihoods of the real predicted tokens (non-pad, and masked in input)
    """

    loss: Float[TT, ""]
    nlls: Float[TT, " num_tokens"]


class MDLMPredictionDict(TypedDict):
    """Prediction results for MDLM.

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
    time_taken: List[float]


TokenLogitsType = Float[TT, " batch seq_len vocab_size"]


class MDLMStepResults(TypedDict):
    """Step results for MDLM.

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
    t: Integer[TT, " batch"]
    change: Optional[Bool[TT, " batch"]]
    constraint: Bool[TT, " batch seq_len"]


class MDLMModel(Protocol):
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
# region: Base MDLM


class MDLMLoss(LossFunction[MDLMBatch, MDLMLossDict]):
    def __init__(
        self,
        loss_on_padding: bool = False,
        model: Optional[MDLMModel] = None,
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
        batch: MDLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MDLMLossDict:
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: MDLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MDLMLossDict:
        x_0, drop, noise_rate, total_noise, t = (
            batch["input_ids"],  # shape (batch, seq_len)
            batch["drop"],  # shape (batch, seq_len)
            batch["noise_rate"],  # shape (batch)
            batch["total_noise"],  # shape (batch)
            batch["t"],  # shape (batch)
        )
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        x_t = torch.where(drop, self.mask_token_id_tensor.view(1, 1), x_0)  # type: ignore
        if self.loss_on_padding:
            model = cast(MDLMModel, self.model)
            logits = model(x_t, total_noise)
        else:
            model = cast(MDLMModel, self.model)
            logits = model(x_t, total_noise, attention_mask)

        # ignore non-mask tokens and pad tokens in loss computation
        non_mask = torch.ne(x_t, self.mask_token_id_tensor.view(1, 1))  # type: ignore
        ignore = (
            non_mask
            if self.loss_on_padding
            else non_mask.logical_or(~attention_mask)
        )
        # TODO (efficiency): The ignore logic can be replaced with
        # masked cross entropy.

        # copy targets because we will be modifying it
        targets = x_0.clone()
        # set the targets for non-mask tokens and pad tokens to -100
        if ignore is not None:
            targets[ignore] = -100

        # Transpose logits to (batch, pred_vocab_size, seq_len) for cross entropy
        logits_T = logits.transpose(1, 2)  # (batch, pred_vocab_size, seq_len)
        ce = torch.nn.functional.cross_entropy(
            logits_T, targets, reduction="none", ignore_index=-100
        )  # (batch, seq_len)

        weight = noise_rate / torch.expm1(total_noise)  # shape (batch, 1)
        kl = ce * weight.unsqueeze(-1)  # (batch, seq_len)
        loss = None
        if ignore.all():
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            loss = kl[~ignore].mean()
        nlls = kl[
            drop.logical_and(attention_mask)
        ]  # collect losses from non-pad masked tokens

        return {
            "loss": loss,
            "nlls": nlls.detach(),
        }


class MDLMPredictor(torch.nn.Module, Predictor[MDLMBatch, MDLMPredictionDict]):
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
        model: Optional[MDLMModel] = None,
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
        self.dt = (1 - 1e-5) / (max_steps + 1)
        self.max_length = max_length
        if sampling_method == "sample":
            self.sampling_function = sample_categorical
        elif sampling_method == "sample_top_k":
            self.sampling_function = partial(sample_from_top_k, top)
        elif sampling_method == "sample_top_p":
            self.sampling_function = partial(sample_from_top_p, p)
        else:
            raise ValueError(f"Invalid sampling method: {sampling_method}")

        self.noise_schedule = noise_schedule
        self.return_history = return_history

    def _predict_single_step(
        self,
        step_results: MDLMStepResults,
        final_step: bool = False,
    ) -> MDLMStepResults:
        """
        Args:
            step_results:
                x: Integer[TT, " batch seq_len"] Current predicted sequence.
                attention_mask: Bool[TT, " batch seq_len"] Mask of the current sequence.
                logits: Float[TT, " batch seq_len vocab_size"] Logits of the current sequence.
                t: Integer[TT, " batch"] Current timestep.
                constraint: Bool[TT, " batch seq_len"] Constraint of the current sequence.
                change: Bool[TT, " batch"] Whether any token in the current sequence is changed.
        """
        # fmt: off
        x_t: Integer[TT, " batch seq_len"] = step_results["x"]
        attention_mask: Bool[TT, " batch seq_len"] = step_results["attention_mask"]
        t: Integer[TT, " batch"] = step_results["t"]
        constraint: Bool[TT, " batch seq_len"] = step_results["constraint"]
        # fmt: on
        s = t - self.dt
        dot_sigma_t: Float[TT, " batch"] = self.noise_schedule(t)[1]
        dot_sigma_s: Float[TT, " batch"] = self.noise_schedule(s)[1]
        # TODO (efficiency): Logits can be cached if the model does not depend on dot_sigma_t
        logits = self.model(x_t, dot_sigma_t, attention_mask)
        chance_s = -torch.expm1(-dot_sigma_s)  # 1 - exp(-dot_sigma_s)
        chance_t = -torch.expm1(-dot_sigma_t)  # 1 - exp(-dot_sigma_t)

        # predicting real tokens
        # TODO (compile): This if is not compile friendly. Split into two functions.
        if not final_step:
            q_xs = torch.softmax(logits, dim=-1) * (
                (chance_t - chance_s)[:, None, None]
            )  # (*batch, seq_len, vocab_size)
            assert (q_xs >= 0).all()
            # predicting mask tokens
            q_xs[:, :, self.tokenizer.mask_token_id] = chance_s[:, None]

            # sanitize: don't predict tokens that are supposed to be in output vocab like pad if loss_on_padding is false
            # q_xs[:, :, self.token_ids_to_suppress] = chance_s[:, None] # a well trained model should not need this.

            # sample
            # xs = sample_categorical(q_xs)  # (*batch, seq_len)
            xs = self.sampling_function(q_xs)  # (*batch, seq_len)
        else:
            q_xs = torch.softmax(
                logits, dim=-1
            )  # (*batch, seq_len, vocab_size)
            xs = torch.argmax(q_xs, dim=-1)  # (*batch, seq_len)

        # copy the input for input positions that were non-mask
        xs = torch.where(
            x_t == self.tokenizer.mask_token_id,
            xs,
            x_t,
        )
        if self.return_history:
            change = (xs != x_t).any(dim=-1)  # shape (batch,)
        else:
            change = None
        return {
            "x": xs,
            "attention_mask": attention_mask,
            "t": s,
            "logits": logits,
            "constraint": constraint,
            "change": change,
        }

    def _stop(
        self,
        step_results: MDLMStepResults,
        current_step: int,
    ) -> bool:
        time_ended = not bool((step_results["t"] > 0).any())
        max_steps_reached = current_step >= self.max_steps
        return time_ended or max_steps_reached

    def decode(self, results: MDLMStepResults) -> Tuple[
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
        step_results: MDLMStepResults,
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
                        float(step_results["t"][batch_idx]),
                        current_step,
                    )
                )
        return history

    @torch._dynamo.disable()
    def predict(
        self,
        batch: MDLMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MDLMPredictionDict:
        _start_time = time.time()
        t_ = batch["t"]
        if t_ is None:
            raise NotImplementedError("Timestep determination not implemented")

        step_results: MDLMStepResults = {
            "x": batch[
                "input_ids"
            ],  # don't clone assuming that the caller is prepared for in-place operations
            "attention_mask": batch["attention_mask"],
            "logits": None,  # type: ignore ok for first step
            "t": t_,
            "change": torch.ones_like(t_, dtype=torch.bool),
            "constraint": None,
        }
        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)
        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(
                step_results,
            )
            history = self._update_history(history, step_results, current_step)
            current_step += 1
        t_final = (t_ - self.dt) * torch.ones_like(t_)
        step_results["t"] = t_final
        step_results = self._predict_single_step(
            step_results,
            final_step=True,
        )
        history = self._update_history(history, step_results, current_step)
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
        ) = self.decode(step_results)

        _end_time = time.time()
        _time_taken = _end_time - _start_time
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "history": history,
            "loss": None,
            "time_taken": [_time_taken]
            * len(out),  # cannot separate time for each sample
        }

    def to_dict(
        self,
        batch: MDLMBatch,  # type: ignore
        preds: MDLMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        preds_list: List[
            Tuple[
                str,
                str,
                List[int],
                List[Tuple[str, float, int]],
                float,
            ]
        ] = list(
            zip(
                preds["text"],
                preds["text_with_spl_tokens"],
                preds["ids"].tolist(),
                preds["history"],
                preds.get(
                    "time_taken", cycle([-1])
                ),  # -1 when the predict method does not measure time.
            )
        )
        dicts: List[Dict[str, Any]] = []
        for text, text_with_spl_tokens, ids, history, time_taken in preds_list:
            rounded_history = [
                [subseq, round(t, 4), step] for subseq, t, step in history
            ]
            dicts.append(
                {
                    "text": text,
                    "text_with_spl_tokens": text_with_spl_tokens,
                    "ids": ids,
                    "history": rounded_history,
                    "time_taken": time_taken,
                }
            )
        return dicts


class RDMPredictor(MDLMPredictor):
    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        tokens_to_suppress: Optional[List[str]] = None,
        return_history: bool = False,
        tau: float = 0.0,
        temp: int = 0.1,
        model: Optional[MDLMModel] = None,
        condition: str = "cond",
        argmax_decoding: bool = False,
        kappa: str = "cosine",
    ):
        super().__init__(
            max_steps=max_steps,
            max_length=max_length,
            tokenizer=tokenizer,
            noise_schedule=noise_schedule,
            tokens_to_suppress=tokens_to_suppress,
            return_history=return_history,
            model=model,
        )
        self.vocab_size = tokenizer.vocab_size
        self.tau = tau
        self.temp = temp
        self.condition = condition
        self.argmax_decoding = argmax_decoding
        self.kappa = kappa

    def _topk_masking(
        self, scores: torch.Tensor, cutoff_len: torch.Tensor, temperature
    ) -> torch.Tensor:
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(scores) + 1e-8) + 1e-8
        )
        noisy_scores = scores + temperature.unsqueeze(-1) * gumbel_noise
        sorted_index = noisy_scores.sort(-1)[0]
        threshold = sorted_index.gather(dim=-1, index=cutoff_len)
        return noisy_scores < threshold

    def _predict_single_step(
        self,
        step_results: MDLMStepResults,
        final_step: bool = False,
    ) -> MDLMStepResults:
        x_t = step_results["x"]
        old_x = x_t.clone()
        output_scores = step_results["scores"]
        attention_mask = step_results["attention_mask"]
        t = step_results["t"]
        constraint = step_results.get(
            "constraint", torch.zeros_like(attention_mask, dtype=torch.bool)
        )
        xt_neq_x0 = step_results["output_masks"]

        noise_rate, dot_sigma_t = self.noise_schedule(t)

        logits = self.model(x_t, dot_sigma_t, attention_mask)

        scores = logits.clone()
        scores[..., self.tokenizer.mask_token_id] = float("-inf")
        scores = torch.log_softmax(scores, dim=-1)

        non_special_sym_mask = ~(constraint.bool())

        if final_step:
            output_scores, x_t = scores.max(-1)
            new_xt_neq_x0 = xt_neq_x0
        else:
            if self.argmax_decoding:
                cur_scores, cur_x = scores.max(-1)
            else:
                cur_x = dists.Categorical(logits=scores / self.temp).sample()
                cur_scores = torch.gather(
                    scores, -1, cur_x.unsqueeze(-1)
                ).squeeze(-1)

            cur_x = torch.where(
                constraint == 1,
                old_x,
                cur_x,
            )
            if self.kappa == "linear":
                rate = 1 - t
            elif self.kappa == "cosine":
                rate = torch.cos(t * torch.pi * 0.5)
            else:
                raise NotImplementedError

            cutoff_len = (
                (non_special_sym_mask.sum(dim=-1).type_as(x_t) * rate)
                .long()
                .reshape(-1, 1)
            )

            scores_for_topk = cur_scores.masked_fill(
                ~(non_special_sym_mask.bool()), float("inf")
            )

            lowest_k_mask = self._topk_masking(
                scores_for_topk, cutoff_len, self.tau * rate
            )

            if self.condition == "cond":
                not_v1_t = (
                    (cur_x == x_t)
                    & (cur_scores < output_scores)
                    & lowest_k_mask
                )
            elif self.condition == "uncond":
                not_v1_t = lowest_k_mask
            else:
                raise NotImplementedError

            not_v2_t = lowest_k_mask

            masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)

            x_t.masked_fill_(masked_to_noise, self.tokenizer.mask_token_id)

            output_scores.masked_fill_(masked_to_noise, float("-inf"))

            masked_to_x0 = xt_neq_x0 & ~not_v2_t
            x_t.masked_scatter_(masked_to_x0, cur_x[masked_to_x0])
            output_scores.masked_scatter_(
                masked_to_x0, cur_scores[masked_to_x0]
            )
            new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t

        x_t = torch.where(
            constraint == 1,
            old_x,
            x_t,
        )

        if self.return_history:
            change = (old_x != x_t).any(dim=-1)
        else:
            change = None

        s = t - self.dt
        return {
            "x": x_t,
            "attention_mask": attention_mask,
            "t": s,
            "logits": logits,
            "constraint": constraint,
            "change": change,
            "scores": output_scores,
            "output_masks": new_xt_neq_x0,
        }

    @torch._dynamo.disable()
    def predict(self, batch: MDLMBatch, *args, **kwargs) -> MDLMPredictionDict:
        _start_time = time.time()
        t_ = batch["t"]
        if t_ is None:
            raise NotImplementedError("Timestep determination not implemented")
        x = batch["input_ids"]
        step_results: MDLMStepResults = {
            "x": x,  # don't clone assuming that the caller is prepared for in-place operations
            "attention_mask": batch["attention_mask"],
            "logits": None,  # type: ignore ok for first step
            "t": t_,
            "change": torch.ones_like(t_, dtype=torch.bool),
            "constraint": batch["constraint"],
            "scores": torch.zeros(
                *x.size(), dtype=torch.float32, device=x.device
            ),
            "output_masks": torch.ones_like(x, dtype=torch.bool),
        }
        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)
        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(
                step_results,
            )
            history = self._update_history(history, step_results, current_step)
            current_step += 1
        t_final = (t_ - self.dt) * torch.ones_like(t_)
        step_results["t"] = t_final
        step_results = self._predict_single_step(
            step_results,
            final_step=True,
        )
        history = self._update_history(history, step_results, current_step)
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
        ) = self.decode(step_results)

        _end_time = time.time()
        _time_taken = _end_time - _start_time
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "history": history,
            "loss": None,
            "time_taken": [_time_taken]
            * len(out),  # cannot separate time for each sample
        }

class ConfidenceBasedPredictor(MDLMPredictor):
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
        model: Optional[MDLMModel] = None,
    ):
        super().__init__(
            max_steps=max_steps,
            max_length=max_length,
            tokenizer=tokenizer,
            noise_schedule=noise_schedule,
            tokens_to_suppress=tokens_to_suppress,
            return_history=return_history,
            model=model,
        )
    def _predict_single_step(
        self,
        step_results: MDLMStepResults,
        final_step: bool = False,
    ) -> MDLMStepResults:
        """
        Args:
            step_results:
                x: Integer[TT, " batch seq_len"] Current predicted sequence.
                attention_mask: Bool[TT, " batch seq_len"] Mask of the current sequence.
                logits: Float[TT, " batch seq_len vocab_size"] Logits of the current sequence.
                t: Integer[TT, " batch"] Current timestep.
                constraint: Bool[TT, " batch seq_len"] Constraint of the current sequence.
                change: Bool[TT, " batch"] Whether any token in the current sequence is changed.
        """
        # fmt: off
        x_t: Integer[TT, " batch seq_len"] = step_results["x"]
        attention_mask: Bool[TT, " batch seq_len"] = step_results["attention_mask"]
        t: Integer[TT, " batch"] = step_results["t"]
        positions = attention_mask.cumsum(dim=1) - 1
        positions *= attention_mask
        constraint: Bool[TT, " batch seq_len"] = step_results["constraint"]
        p_x0 = step_results["p_x0"]
        conf:  TokenLogitsType = step_results["conf"]
        # fmt: on
        s = t - self.dt
        dot_sigma_t: Float[TT, " batch"] = self.noise_schedule(t)[1]
        dot_sigma_s: Float[TT, " batch"] = self.noise_schedule(s)[1]
        # TODO (efficiency): Logits can be cached if the model does not depend on dot_sigma_t
        logits = self.model(x_t, dot_sigma_t, attention_mask)
        if p_x0 is None:
            p_x0 = logits.exp()
        if conf is None:
           conf = - torch.ones_like(x_t, dtype=p_x0.dtype) * torch.inf

        if not final_step:
            chance_t = t[:, None, None] #test vanilla mdlm form
            chance_s = s[:, None, None]
            alpha_t = (1 - chance_t)[0].item()
            alpha_s = (1 - chance_s)[0].item()
            
            if alpha_t > 0:
                sigma_max = min(1, (1 - alpha_s) / alpha_t)
            else:
                sigma_max = 1
            eta = conf.softmax(dim=-1)
            masked_flag = (x_t == self.tokenizer.mask_token_id).to(torch.bool)
            eta[masked_flag] = 0
            sigma = eta * sigma_max
            q_xs = p_x0 * (1 - sigma[:, :, None])
            q_xs[..., self.tokenizer.mask_token_id] = sigma
            q_xs_2 = p_x0 * ((alpha_s - (1 - sigma[:, :, None]) * alpha_t) / (1 - alpha_t))
            q_xs_2[..., self.tokenizer.mask_token_id] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
            copy_flag = (x_t != self.tokenizer.mask_token_id).to(torch.bool)
            q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
            xs = self.sampling_function(q_xs)
            unmask_mask = (x_t == self.tokenizer.mask_token_id) & (xs != self.tokenizer.mask_token_id)
            batch_indices = torch.arange(xs.shape[0])[:, None]
            feature_indices = torch.arange(xs.shape[1])
            conf_values = -p_x0[batch_indices, feature_indices, xs]
            conf[unmask_mask] = conf_values[unmask_mask]
            remask_mask = (x_t != self.tokenizer.mask_token_id) & (xs == self.tokenizer.mask_token_id)
            conf[remask_mask] = -torch.inf
        else:
            q_xs = torch.softmax(
                logits, dim=-1
            )
            xs = torch.argmax(q_xs, dim=-1)

        xs = torch.where(
            x_t == self.tokenizer.mask_token_id,
            xs,
            x_t,
        )
        if self.return_history:
            change = (xs != x_t).any(dim=-1)  # shape (batch,)
        else:
            change = None
        return {
            "x": xs,
            "attention_mask": attention_mask,
            "t": s,
            "logits": logits,
            "constraint": constraint,
            "change": change,
            "conf": conf,
            "p_x0": p_x0
        }
    
    
    @torch._dynamo.disable()
    def predict(
        self,
        batch: MDLMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MDLMPredictionDict:
        _start_time = time.time()
        t_ = batch["t"]
        if t_ is None:
            raise NotImplementedError("Timestep determination not implemented")
        x = batch["input_ids"]
        step_results: MDLMStepResults = {
            "x": x,  # don't clone assuming that the caller is prepared for in-place operations
            "attention_mask": batch["attention_mask"],
            "logits": None,  # type: ignore ok for first step
            "t": t_,
            "change": torch.ones_like(t_, dtype=torch.bool),
            "constraint": None,
            "conf": None,
            "p_x0": None
        }
        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)
        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(
                step_results,
            )
            history = self._update_history(history, step_results, current_step)
            current_step += 1
        t_final = (t_ - self.dt) * torch.ones_like(t_)
        step_results["t"] = t_final
        step_results = self._predict_single_step(
            step_results,
            final_step=True,
        )
        history = self._update_history(history, step_results, current_step)
        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
        ) = self.decode(step_results)

        _end_time = time.time()
        _time_taken = _end_time - _start_time
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "history": history,
            "loss": None,
            "time_taken": [_time_taken]
            * len(out),  # cannot separate time for each sample
        }


class MDLMPathPlanningPredictor(MDLMPredictor):
    """
    Path Planning implementation of the MDLMPredictor.
    Overrides only the necessary methods to implement the path planning algorithm.
    """

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        tokens_to_suppress: Optional[List[str]] = None,
        return_history: bool = False,
        sampling_method: str = "path_planning",
        top: int = 1000,
        p: float = 0.9,
        # Path planning specific parameters
        tau: float = 1.0,
        eta: float = 1.0,
        kappa_fn: Callable[[float], float] = lambda t: t,
        score_type: str = "confidence",
        planner: Optional[torch.nn.Module] = None,
        model: Optional[MDLMModel] = None,
    ):
        # Initialize the parent class with standard parameters
        super().__init__(
            max_steps=max_steps,
            max_length=max_length,
            tokenizer=tokenizer,
            noise_schedule=noise_schedule,
            tokens_to_suppress=tokens_to_suppress,
            return_history=return_history,
            model=model,
            sampling_method="sample",
            top=top,
            p=p,
        )

        # Store path planning specific parameters
        self.tau = tau
        self.eta = eta
        self.kappa_fn = kappa_fn
        self.score_type = score_type
        if planner is not None:
            self.planner = planner
        else:
            self.planner = model

    def _topk_lowest_masking(self, scores, cutoff_len):
        """Helper function to identify tokens with lowest scores to mask"""
        sorted_scores, _ = scores.sort(dim=-1)
        threshold = sorted_scores.gather(dim=-1, index=cutoff_len)
        return scores < threshold

    def _stochastic_sample_from_categorical(
        self, logits, temperature=1.0, noise_scale=1.0
    ):
        """Helper function for path planning sampling with temperature and noise"""
        logits = logits.double()
        if temperature != 0.0:
            gumbel = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-8) + 1e-8
            )
            logits = logits / temperature + noise_scale * gumbel
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
        return tokens, scores

    def _predict_single_step(
        self,
        step_results: MDLMStepResults,
        final_step: bool = False,
    ) -> MDLMStepResults:
        """
        Override the parent class method with path planning implementation
        """
        x_t = step_results["x"]
        attention_mask = step_results["attention_mask"]
        t = step_results["t"]
        constraint = step_results.get(
            "constraint", torch.zeros_like(attention_mask, dtype=torch.bool)
        )

        # Calculate current time in [0,1] range
        step_num = int((1 - t[0].item()) / self.dt)
        t_normalized = step_num * self.dt

        # Fix mask identifies tokens that should never be changed
        fix_mask = x_t != self.tokenizer.mask_token_id

        # Get model predictions
        dot_sigma_t: Float[TT, " batch"] = self.noise_schedule(t)[1]
        logits = self.model(x_t, dot_sigma_t, attention_mask).double()

        # Track which tokens are currently masked
        last_mask = x_t == self.tokenizer.mask_token_id

        # Identify which tokens can be potentially unmasked
        unmask_candidates = ~last_mask & ~fix_mask

        # Sample tokens based on logits
        x0, logp = self._stochastic_sample_from_categorical(
            logits, temperature=self.tau
        )

        # Incorporate planner if available
        if self.planner is not None:
            with torch.inference_mode():
                planner_logits = self.planner(x0).double()
                planner_logp = (
                    planner_logits.log_softmax(dim=-1)
                    .gather(-1, x0.unsqueeze(-1))
                    .squeeze(-1)
                )
                logits[unmask_candidates] = planner_logits[unmask_candidates]
                logp[unmask_candidates] = planner_logp[unmask_candidates]

        # Compute token scores for masking decision
        if self.score_type == "confidence":
            score = logp
        elif self.score_type == "random":
            score = torch.rand_like(logp).log()
        else:
            raise ValueError(f"Invalid score_type: {self.score_type}")

        # Ensure fixed tokens are never masked
        score = score.masked_fill(fix_mask, float("inf"))

        # Apply scaling factor to unmask candidates
        score[unmask_candidates] *= self.eta

        # Calculate how many tokens to mask at this step
        kappa_t = self.kappa_fn(t_normalized)
        num_to_mask = (
            (~fix_mask).sum(dim=1, keepdim=True).float() * (1 - kappa_t)
        ).long()

        # Get mask for tokens with lowest scores
        mask = self._topk_lowest_masking(score, num_to_mask)

        # Apply masking and unmasking operations
        new_x = x_t.clone()
        new_x[mask] = self.tokenizer.mask_token_id

        # Unmask tokens that were masked before but shouldn't be masked now
        mask_to_x0 = last_mask & ~mask
        new_x[mask_to_x0] = x0[mask_to_x0]

        # For final step, replace all remaining masks with predictions
        if final_step:
            remaining_mask = new_x == self.tokenizer.mask_token_id
            new_x[remaining_mask] = x0[remaining_mask]

        # Calculate if any tokens have changed
        if self.return_history:
            change = (new_x != x_t).any(dim=-1)  # shape (batch,)
        else:
            change = None

        # Update timestep
        s = t - self.dt

        return {
            "x": new_x,
            "attention_mask": attention_mask,
            "t": s,
            "logits": logits,
            "constraint": constraint,
            "change": change,
        }

    @torch._dynamo.disable()
    def predict(
        self,
        batch: MDLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MDLMPredictionDict:
        """
        Override the parent's predict method with path planning specific handling

        This is mostly the same as the parent class but ensures constraints
        are properly initialized.
        """

        _start_time = time.time()
        t_ = batch["t"]
        if t_ is None:
            raise NotImplementedError("Timestep determination not implemented")

        step_results: MDLMStepResults = {
            "x": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "logits": None,  # type: ignore ok for first step
            "t": t_,
            "change": torch.ones_like(t_, dtype=torch.bool),
            "constraint": batch.get(
                "constraint",
                torch.zeros_like(batch["attention_mask"], dtype=torch.bool),
            ),
        }
        current_step = 1
        history: List[List[Tuple[str, float, int]]] = [
            [] for _ in range(batch["input_ids"].shape[0])
        ]
        history = self._update_history(history, step_results, current_step)

        while not self._stop(step_results, current_step):
            step_results = self._predict_single_step(
                step_results,
            )
            history = self._update_history(history, step_results, current_step)
            current_step += 1

        t_final = (t_ - self.dt) * torch.ones_like(t_)
        step_results["t"] = t_final
        step_results = self._predict_single_step(
            step_results,
            final_step=True,
        )
        history = self._update_history(history, step_results, current_step)

        # decode the final step
        (
            out,
            out_with_spl_tokens,
            final_x,
        ) = self.decode(step_results)

        _end_time = time.time()
        _time_taken = _end_time - _start_time
        return {
            "text": out,
            "text_with_spl_tokens": out_with_spl_tokens,
            "ids": final_x,
            "history": history,
            "loss": None,
            "time_taken": [_time_taken] * len(out),
        }


class NLL(MeanMetric):
    pass


class Perplexity(NLL):
    def compute(self):
        return torch.exp(self.mean_value / self.weight)


class MDLMLightningModule(BaseLightningModule):
    predictor: MDLMPathPlanningPredictor
    loss_function: MDLMLoss

    def setup_predictor(self):
        self.predictor = hydra.utils.instantiate(
            self.config.predictor,
            tokenizer=self.tokenizer,
            noise_schedule=self.noise_schedule,
        )

    def check_loss_predictor_consistency(self):
        if hasattr(self.loss_function, "loss_on_padding"):
            if hasattr(self.predictor, "token_ids_to_suppress"):
                # make sure pad_token_id is not in token_ids_to_suppress
                if self.loss_function.loss_on_padding:
                    assert (
                        self.tokenizer.pad_token_id
                        not in self.predictor.token_ids_to_suppress
                    )
                else:
                    assert (
                        self.tokenizer.pad_token_id
                        in self.predictor.token_ids_to_suppress
                    )

    def setup_metrics(self):
        # Initialize diagnostic metrics
        nll_perplexity = MetricCollection(
            {"nll": NLL(), "perplexity": Perplexity()}
        )
        self.train_nll_perplexity = nll_perplexity.clone(prefix="train/")
        self.val_nll_perplexity = nll_perplexity.clone(prefix="val/")
        self.test_nll_perplexity = nll_perplexity.clone(prefix="test/")

    def _prepare_input_batch_for_predict(self, batch: MDLMBatch) -> MDLMBatch:
        """Use the `drop` tensor in batch to update `input_ids`.

        `input_ids` is set to the mask token for all drop positions. Use this function on a batch
        if the batch contains ground truth ids but the `drop` is set.

        Note:
            This function will do nothing if the `drop` is not set or if the `drop` tensor is all zeros.
        """
        # clone the batch assuming that the predictor will do in-place operations on the batch
        # TODO (efficiency): Only need to clone `input_ids`.
        cloned_batch: MDLMBatch = {}  # type: ignore
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                cloned_batch[k] = v.clone()
            else:
                cloned_batch[k] = v
        # TODO (compile): tokenizer.mask_token_id will be int.
        cloned_batch["input_ids"][batch["drop"]] = self.tokenizer.mask_token_id
        return cloned_batch

    def compute_loss(
        self,
        batch: MDLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Optional[Union[MDLMLossDict, MDLMPredictionDict]]:
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
                cloned_batch, batch_idx, dataloader_idx, dataloader_name
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


# endregion: Base MDLM
################################################################################


################################################################################
# region: MDLM for Star Graphs


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


class MDLMLightningModuleForStarGraphs(MDLMLightningModule):
    def _get_generated_length(
        self, pred_ids: Integer[TT, " batch seq_len"]
    ) -> Integer[TT, " batch"]:
        pad_token_id = int(cast(int, self.tokenizer.pad_token_id))
        mask_token_id = int(cast(int, self.tokenizer.mask_token_id))
        remove = torch.tensor(
            [mask_token_id, pad_token_id], device=pred_ids.device
        )
        return torch.isin(pred_ids, remove, invert=True).sum(dim=-1)

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
            actual_length = batch["attention_mask"].sum(dim=-1)

            self.val_generated_length_rmse.update(
                generated_length, actual_length
            )
            self.val_generated_length_mape.update(
                generated_length, actual_length
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
        batch: MDLMBatch,
        preds: MDLMPredictionDict,
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


# endregion: MDLM for Star Graphs
################################################################################

################################################################################
# region: MDLM for Zebra

MDLMLightningModuleForZebra = MDLMLightningModuleForStarGraphs

# endregion: MDLM for Zebra
################################################################################
