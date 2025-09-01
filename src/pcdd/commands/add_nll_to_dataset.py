import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Generator, Optional

import dotenv
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pcdd.utils import omegaconf_resolvers
from pcdd.utils.rich_utils import print_config_tree


dotenv.load_dotenv()
logger = logging.getLogger(__name__)


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path("../../../configs") / "lightning_train"),
    "config_name": "dataset_nll.yaml",
}

# Register a temporary resolver so that early calls to resolve() don't fail
omegaconf_resolvers.register_resolvers()
OmegaConf.register_new_resolver(
    "datamodule", lambda attr: "${datamodule:" + str(attr) + "}"
)
OmegaConf.register_new_resolver(
    "tokenizer", lambda attr: "${tokenizer:" + str(attr) + "}"
)
OmegaConf.register_new_resolver(
    "lightning_module", lambda attr: "${lightning_module:" + str(attr) + "}"
)
# endregion


def strings(dataset) -> Generator[str, None, None]:
    for example in dataset:
        yield example["text"]


def get_wandb(cfg: DictConfig) -> Optional[Any]:
    if loggers := cfg.get("loggers"):
        for name, logger in loggers.items():
            if name == "wandb":
                import wandb

                cfg_to_log = OmegaConf.to_container(cfg, resolve=True)
                run = wandb.init(
                    entity=logger.get("entity"),
                    name=logger.get("name"),
                    id=logger.get("id"),
                    project=logger.get("project"),
                    job_type=logger.get("job_type"),
                    tags=logger.get("tags"),
                    config=cfg_to_log,
                )
                return run
    return None


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    from lightning_utilities.core.rank_zero import rank_zero_only

    rank_zero_only.rank = 0
    print_config_tree(cfg, resolve=True, save_to_file=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = hydra.utils.instantiate(cfg.evaluator.model).to(device)
    name = cfg.evaluator.name
    tokenizer = hydra.utils.instantiate(cfg.evaluator.tokenizer)
    dataset = hydra.utils.instantiate(cfg.dataset)
    num_examples = cfg.get("num_examples") or len(dataset)
    print(f"Processing {num_examples} examples.")
    dataset = dataset.select(range(num_examples))
    wandb_run = get_wandb(cfg)
    data_dir = Path(cfg.paths.data_dir)
    dataset_str = cfg.dataset.get("path")
    if not dataset_str:
        dataset_str = cfg.dataset.get("name")

    output_file = cfg.get("output_file")
    if output_file is None:
        # try
        if not dataset_str:
            raise ValueError("No dataset name or path provided")
        output_file = data_dir / dataset_str / "nll.jsonl"
    logger.info(f"Writing NLLs to {output_file}")
    samples_file = cfg.get("output_samples_file")
    if samples_file is None:
        # try
        if not dataset_str:
            raise ValueError("No dataset name or path provided")
        samples_file = data_dir / dataset_str / "samples.jsonl"
    logger.info(f"Writing samples to {samples_file}")

    nlls = []
    entropies = []
    lengths = []
    with open(samples_file, "w") as f:
        with torch.no_grad():
            for i, string in enumerate(strings(dataset)):
                inputs = tokenizer([string], return_tensors="pt")
                inputs.to(device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16
                ):
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
                f.write(
                    json.dumps(
                        {
                            "nll": nll,
                            "entropy": entropy,
                            "length": length,
                        }
                    )
                    + "\n"
                )
                if i % 100 == 0:
                    print(f"Processed {i}/{num_examples} examples")

    print(f"Processed {i}/{num_examples} examples")

    nlls = torch.tensor(nlls)
    entropies = torch.tensor(entropies)
    lengths = torch.tensor(lengths).to(dtype=torch.float32)

    print(
        f"Model: {name}, NLL: {nlls.mean()}, Entropy: {entropies.mean()}, Length: {lengths.mean()}"
    )
    if wandb_run is not None:
        wandb_run.summary["nll"] = nlls.mean().item()
        wandb_run.summary["entropy"] = entropies.mean().item()
        wandb_run.summary["length"] = lengths.mean().item()
        wandb_run.summary["num_examples"] = num_examples
        wandb_run.summary["model"] = name

    if (output_file := cfg.get("output_file")) is not None:
        logger.info(f"Writing metrics to {output_file}")
        with open(output_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "nll": nlls.mean().item(),
                        "entropy": entropies.mean().item(),
                        "model": name,
                        "length": lengths.mean().item(),
                        "num_examples": num_examples,
                    }
                )
                + "\n"
            )
    if wandb_run is not None:
        wandb_run.finish()


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
