from collections import Counter
import json
from typing import Dict, Generator, List, Optional
from omegaconf import DictConfig
import torch
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import logging
from pcdd.diffusion.generative_perplexity import (
    GenerativePerplexityEvaluatorResult,
)
from pcdd.diffusion.lightning_module_v2 import GenerativePerplexityCallback
from more_itertools import batched
from torchmetrics import Perplexity
import dotenv

from pcdd.utils.nn import masked_mean, masked_sum

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path("../../../configs") / "lightning_train"),
    "config_name": "generative_perplexity.yaml",
}


def strings(file_path: Path) -> Generator[str, None, None]:
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            yield data["text"]


def compute_metrics(
    logits: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    if logits.shape[0] != 1:
        raise ValueError("logits must have shape (1, seq_len, vocab_size)")


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = hydra.utils.instantiate(cfg.evaluator.model).to(device)
    name = cfg.evaluator.name
    tokenizer = hydra.utils.instantiate(cfg.evaluator.tokenizer)
    batch_size = cfg.evaluator.batch_size

    nlls = []
    entropies = []
    lengths = []
    with torch.no_grad():
        for string in strings(cfg.samples_file):
            inputs = tokenizer([string], return_tensors="pt")
            inputs.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = evaluator(**inputs)
            all_log_probs = torch.log_softmax(outputs.logits, dim=-1)[
                0, :-1, :
            ]
            log_p = all_log_probs[
                torch.arange(
                    all_log_probs.shape[0], device=all_log_probs.device
                ),
                inputs.input_ids[0, 1:],
            ]
            length = inputs.input_ids.shape[1] - 1
            nll = -log_p.mean().item()
            counts = torch.tensor(
                list(Counter(inputs.input_ids[0, 1:].tolist()).values())
            )
            total = counts.sum()
            p_for_entropy = counts / total
            entropy = -torch.sum(
                p_for_entropy * torch.log(p_for_entropy)
            ).item()
            nlls.append(nll)
            entropies.append(entropy)
            lengths.append(length)

    nlls = torch.tensor(nlls)
    entropies = torch.tensor(entropies)
    lengths = torch.tensor(lengths).to(dtype=torch.float32)

    print(
        f"Model: {name}, NLL: {nlls.mean()}, Entropy: {entropies.mean()}, Length: {lengths.mean()}"
    )
    samples_file = Path(cfg.samples_file)
    samples_file_name = samples_file.name
    file_ = samples_file.parent / f"{samples_file_name}.metrics.jsonl"
    print(f"Writing metrics to {file_}")
    with open(file_, "w") as f:
        f.write(
            json.dumps(
                {
                    "nll": nlls.mean().item(),
                    "entropy": entropies.mean().item(),
                    "model": name,
                    "length": lengths.mean().item(),
                }
            )
        )


# @hydra.main(**_HYDRA_PARAMS)
# def main(cfg: DictConfig) -> None:
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    evaluator = hydra.utils.instantiate(cfg.evaluator.model).to(device)
#    name = cfg.evaluator.name
#    tokenizer = hydra.utils.instantiate(cfg.evaluator.tokenizer)
#    if tokenizer.pad_token is None:
#        tokenizer.pad_token = tokenizer.eos_token
#    batch_size = cfg.evaluator.batch_size
#    batch_size = 8
#
#    nlls = []
#    entropies = []
#    num_examples = 0
#    with torch.no_grad():
#        for _strings in batched(strings(cfg.samples_file), batch_size):
#            batch_size = len(_strings)
#            inputs = tokenizer(
#                _strings, return_tensors="pt", truncation=True, padding=True
#            )
#            inputs.to(device)
#            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
#                outputs = evaluator(**inputs)
#            all_log_probs = torch.log_softmax(outputs.logits, dim=-1)[
#                :, :-1, :
#            ]
#            log_p = all_log_probs[
#                torch.arange(
#                    all_log_probs.shape[0], device=all_log_probs.device
#                ).unsqueeze(-1),
#                inputs.input_ids[:, 1:],
#            ]
#            mask = inputs.attention_mask[:, 1:]
#            nll = -masked_mean(log_p, mask, dim=-1)  # (batch_size,)
#            counts = torch.nn.functional.one_hot(inputs.input_ids[:, 1:]).sum(
#                dim=1
#            )  # (batch_size, vocab_size)
#            total = counts.sum(dim=-1)
#            p_for_entropy = counts / total
#            entropy = -masked_sum(
#                p_for_entropy * torch.log(p_for_entropy), mask, dim=-1
#            )  # (batch_size,)
#            nlls.append(nll.sum().item())
#            entropies.append(entropy.sum().item())
#            num_examples += batch_size
#
#    nlls = sum(nlls) / num_examples
#    entropies = sum(entropies) / num_examples
#    print(f"Model: {name}, NLL: {nlls}, Entropy: {entropies}")


if __name__ == "__main__":
    main()
