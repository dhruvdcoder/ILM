"""Similar to compute_generative_preplexity.py, but adds NLL to the samples file itself. Useful for post analysis that requires NLL per sample."""

from collections import Counter
import json
from typing import Any, Dict, Generator, Set
from omegaconf import DictConfig
import torch
from pathlib import Path
import hydra
import logging
import dotenv


dotenv.load_dotenv()
logger = logging.getLogger(__name__)


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path("../../../configs") / "lightning_train"),
    "config_name": "generative_perplexity.yaml",
}


def all_samples(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            yield data


def get_stats(device, evaluator, inputs):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        outputs = evaluator(**inputs)
    all_log_probs = torch.log_softmax(outputs.logits, dim=-1)[0, :-1, :]
    log_p = all_log_probs[
        torch.arange(all_log_probs.shape[0], device=all_log_probs.device),
        inputs.input_ids[0, 1:],
    ]
    length = inputs.input_ids.shape[1] - 1
    nll = -log_p.mean().item()
    counts = torch.tensor(
        list(Counter(inputs.input_ids[0, 1:].tolist()).values())
    )
    total = counts.sum()
    p_for_entropy = counts / total
    entropy = -torch.sum(p_for_entropy * torch.log(p_for_entropy)).item()
    return length, nll, entropy


SPECIAL_TOKENS = {
    "[SEP]",
    "[MASK]",
    "[BOS]",
    "[EOS]",
    "[PAD]",
    "[UNK]",
    "[CLS]",
}


def remove_special_tokens(
    text: str, special_tokens: Set[str] = SPECIAL_TOKENS
) -> str:
    return " ".join(
        [tok for tok in text.split(" ") if tok not in SPECIAL_TOKENS]
    )


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = hydra.utils.instantiate(cfg.evaluator.model).to(device)
    name = cfg.evaluator.name
    tokenizer = hydra.utils.instantiate(cfg.evaluator.tokenizer)
    samples_file = Path(cfg.samples_file)
    samples_file_name = samples_file.name
    output_file = (
        samples_file.parent / f"{samples_file_name}.samples_with_metrics.jsonl"
    )
    logger.info(f"Writing per sample metrics to {output_file}")
    file_ = samples_file.parent / f"{samples_file_name}.only_metrics.jsonl"

    generated_nlls = []
    generated_entropies = []
    generated_lengths = []
    input_nlls = []
    input_entropies = []
    input_lengths = []
    try:
        with torch.no_grad():
            with open(output_file, "w") as f:
                for i, _sample in enumerate(all_samples(cfg.samples_file)):
                    generated_str = _sample["text"]
                    input_str = remove_special_tokens(_sample["input_text"])

                    gen_inputs = tokenizer(
                        [generated_str], return_tensors="pt"
                    )
                    gen_inputs.to(device)
                    inp_inputs = tokenizer(
                        [input_str],
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    inp_inputs.to(device)

                    gen_length, gen_nll, gen_entropy = get_stats(
                        device, evaluator, gen_inputs
                    )
                    inp_length, inp_nll, inp_entropy = get_stats(
                        device, evaluator, inp_inputs
                    )

                    pct_nll = (
                        ((gen_nll - inp_nll) / inp_nll * 100)
                        if inp_nll != 0
                        else 0
                    )
                    pct_entropy = (
                        ((gen_entropy - inp_entropy) / inp_entropy * 100)
                        if inp_entropy != 0
                        else 0
                    )
                    pct_length = (
                        ((gen_length - inp_length) / inp_length * 100)
                        if inp_length != 0
                        else 0
                    )

                    generated_nlls.append(gen_nll)
                    generated_entropies.append(gen_entropy)
                    generated_lengths.append(gen_length)
                    input_nlls.append(inp_nll)
                    input_entropies.append(inp_entropy)
                    input_lengths.append(inp_length)

                    f.write(
                        json.dumps(
                            {
                                "generated_nll": gen_nll,
                                "generated_entropy": gen_entropy,
                                "generated_length": gen_length,
                                "input_nll": inp_nll,
                                "input_entropy": inp_entropy,
                                "input_length": inp_length,
                                "pct_change_nll": pct_nll,
                                "pct_change_entropy": pct_entropy,
                                "pct_change_length": pct_length,
                                **_sample,
                            }
                        )
                        + "\n",
                    )
                    if i % 100 == 0:
                        logger.info(f"Processed {i} samples")

    except Exception as e:
        logger.error(f"Error processing sample {i}: {e}")
        raise e
    finally:
        logger.info(f"Processed {i} samples")
        generated_nlls_tensor = torch.tensor(generated_nlls)
        generated_entropies_tensor = torch.tensor(generated_entropies)
        generated_lengths_tensor = torch.tensor(
            generated_lengths, dtype=torch.float32
        )
        input_nlls_tensor = torch.tensor(input_nlls)
        input_entropies_tensor = torch.tensor(input_entropies)
        input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.float32)

        mean_generated_nll = generated_nlls_tensor.mean().item()
        mean_generated_entropy = generated_entropies_tensor.mean().item()
        mean_generated_length = generated_lengths_tensor.mean().item()
        mean_input_nll = input_nlls_tensor.mean().item()
        mean_input_entropy = input_entropies_tensor.mean().item()
        mean_input_length = input_lengths_tensor.mean().item()

        mean_pct_nll = (
            (
                (
                    (generated_nlls_tensor - input_nlls_tensor)
                    / input_nlls_tensor
                )
                * 100
            )
            .mean()
            .item()
        )
        mean_pct_entropy = (
            (
                (
                    (generated_entropies_tensor - input_entropies_tensor)
                    / input_entropies_tensor
                )
                * 100
            )
            .mean()
            .item()
        )
        mean_pct_length = (
            (
                (
                    (generated_lengths_tensor - input_lengths_tensor)
                    / input_lengths_tensor
                )
                * 100
            )
            .mean()
            .item()
        )

        print(
            f"Model: {name}, Generated NLL: {mean_generated_nll}, Input NLL: {mean_input_nll}, Percent Change NLL: {mean_pct_nll}"
        )
        print(
            f"Model: {name}, Generated Entropy: {mean_generated_entropy}, Input Entropy: {mean_input_entropy}, Percent Change Entropy: {mean_pct_entropy}"
        )
        print(
            f"Model: {name}, Generated Length: {mean_generated_length}, Input Length: {mean_input_length}, Percent Change Length: {mean_pct_length}"
        )

        print(f"Writing aggregated metrics to {file_}")
        with open(file_, "w") as f:
            f.write(
                json.dumps(
                    {
                        "generated": {
                            "nll": mean_generated_nll,
                            "entropy": mean_generated_entropy,
                            "length": mean_generated_length,
                        },
                        "input": {
                            "nll": mean_input_nll,
                            "entropy": mean_input_entropy,
                            "length": mean_input_length,
                        },
                        "percent_change": {
                            "nll": mean_pct_nll,
                            "entropy": mean_pct_entropy,
                            "length": mean_pct_length,
                        },
                        "model": name,
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
