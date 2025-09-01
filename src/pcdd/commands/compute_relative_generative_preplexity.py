from collections import Counter
import json
from typing import Dict, Generator, List, Optional, Tuple
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
from tabulate import tabulate

from pcdd.utils.nn import masked_mean, masked_sum

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path("../../../configs") / "lightning_train"),
    "config_name": "generative_perplexity.yaml",
}


def output_input(file_path: Path) -> Generator[Tuple[str, str], None, None]:
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            yield data["text"], data["input_text"]


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

    input_nlls = []
    input_entropies = []
    input_lengths = []
    generated_nlls = []
    generated_entropies = []
    generated_lengths = []
    with torch.no_grad():
        for generated, inp in output_input(cfg.samples_file):
            generated_tokens = tokenizer([generated], return_tensors="pt")
            generated_tokens.to(device)
            input_tokens = tokenizer(
                [inp],
                return_tensors="pt",
                skip_special_tokens=True,  # we dont' want [MASK] to be included.
            )
            input_tokens.to(device)

            length, nll, entropy = get_stats(
                device, evaluator, generated_tokens
            )
            generated_nlls.append(nll)
            generated_entropies.append(entropy)
            generated_lengths.append(length)

            length, nll, entropy = get_stats(device, evaluator, input_tokens)
            input_nlls.append(nll)
            input_entropies.append(entropy)
            input_lengths.append(length)

    generated_nlls = torch.tensor(generated_nlls)
    mean_generated_nlls = generated_nlls.mean()
    generated_entropies = torch.tensor(generated_entropies)
    mean_generated_entropies = generated_entropies.mean()
    generated_lengths = torch.tensor(generated_lengths).to(dtype=torch.float32)
    mean_generated_lengths = generated_lengths.mean()
    input_nlls = torch.tensor(input_nlls)
    mean_input_nlls = input_nlls.mean()
    input_entropies = torch.tensor(input_entropies)
    mean_input_entropies = input_entropies.mean()
    input_lengths = torch.tensor(input_lengths).to(dtype=torch.float32)
    mean_input_lengths = input_lengths.mean()
    # percent change in nll
    mean_percent_change_in_nll = (
        ((generated_nlls - input_nlls) / input_nlls * 100).mean().item()
    )
    # percent change in entropy
    mean_percent_change_in_entropy = (
        ((generated_entropies - input_entropies) / input_entropies * 100)
        .mean()
        .item()
    )
    # percent change in length
    mean_percent_change_in_length = (
        ((generated_lengths - input_lengths) / input_lengths * 100)
        .mean()
        .item()
    )

    # Prepare data for the table
    table_data = [
        ["Metric", "Generated", "Input", "Percent Change"],
        [
            "NLL",
            mean_generated_nlls,
            mean_input_nlls,
            mean_percent_change_in_nll,
        ],
        [
            "Entropy",
            mean_generated_entropies,
            mean_input_entropies,
            mean_percent_change_in_entropy,
        ],
        [
            "Length",
            mean_generated_lengths,
            mean_input_lengths,
            mean_percent_change_in_length,
        ],
    ]

    # Print the table
    print(f"Model: {name}")
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

    samples_file = Path(cfg.samples_file)
    samples_file_name = samples_file.name
    file_ = samples_file.parent / f"{samples_file_name}.metrics.jsonl"
    print(f"Writing metrics to {file_}")
    with open(file_, "w") as f:
        f.write(
            json.dumps(
                {
                    "nll": mean_generated_nlls,
                    "entropy": mean_generated_entropies,
                    "model": name,
                    "length": mean_generated_lengths,
                }
            )
        )


def get_stats(device, evaluator, inputs_generated):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        outputs_generated = evaluator(**inputs_generated)

    all_log_probs = torch.log_softmax(outputs_generated.logits, dim=-1)[
        0, :-1, :
    ]
    log_p = all_log_probs[
        torch.arange(all_log_probs.shape[0], device=all_log_probs.device),
        inputs_generated.input_ids[0, 1:],
    ]
    length = inputs_generated.input_ids.shape[1] - 1
    nll = -log_p.mean().item()
    counts = torch.tensor(
        list(Counter(inputs_generated.input_ids[0, 1:].tolist()).values())
    )
    total = counts.sum()
    p_for_entropy = counts / total
    entropy = -torch.sum(p_for_entropy * torch.log(p_for_entropy)).item()

    return length, nll, entropy


if __name__ == "__main__":
    main()
