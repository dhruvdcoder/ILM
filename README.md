
# Setup

1. Create a new environment using conda.

```
conda create -p ./.venv_text_diffusion python=3.11.10 pip ipykernel -y
conda activate ./.venv_text_diffusion
```

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
pip install -r core_requirements.txt && \
pip install -r test_requirements.txt && \
pip install -r lint_requirements.txt && \
pip install -e .
```

2. Create a `.env` file in the root directory with the following content:

```
DATA_DIR=/path/to/the/directory/to/store/data
WANDB_ENTITY=<wandb_username>
WANDB_PROJECT=<wandb_project_name>
TOKENIZERS_PARALLELISM=false
RESULTS_DIR=/path/to/directory/to/store/results
PROJECT_ROOT=.
HYDRA_FULL_ERROR=1
```
If you do not plan to use wandb for logging, the `WANDB_ENTITY` and `WANDB_PROJECT` will be ignored.

> [!NOTE]
> The `star_small`, `vstar_small` and `vstar_medium` in the code correspond to the easy, medium and hard versions of the star graph datasets in the paper. 

 ILM = Insertion Language Model, IT = Insertion Transformer[^1],  XLNet = XLNet[^2], MDM = Masked Diffusion Model[^3], ARM = Autoregressive (left-to-right) Model

# Training

The scripts listed in the table below are present in the `scripts/` directory.

| Model | Dataset       | Script                                | Weights                | Experiment                   |
|-------|---------------|---------------------------------------|------------------------|------------------------------|
| ILM   | Star (easy)   | `train_ilm_stopping_star_small.sh`    | -                      | X                            |
| ILM   | Star (medium) | `train_ilm_stopping_vstar_small.sh`   | `ilm-star_medium.pt`   | X                            |
| ILM   | Star (hard)   | `train_ilm_stopping_vstar_medium.sh`  | `ilm-star_hard.pt`     | X                            |
| ---   | ---           | ---                                   | ---                    | ---                          |
| MDM   | Star (easy)   | `train_mdlm_star_small.sh`            | `mdm-star_easy.pt`     | `mdlm_star_small`            |
| MDM   | Star (medium) | `train_mdlm_vstar_small.sh`           | `mdm-star_medium.pt`   | `mdlm_vstar_small`           |
| MDM   | Star (hard)   | `train_mdlm_vstar_medium.sh`          | `mdm-star_hard.pt`     | `mdlm_vstar_medium`          |
| ---   | ---           | ---                                   | ---                    | ---                          |
| ARM   | Star (easy)   | `train_arlm_star_small.sh`            | -                      | X                            |
| ARM   | Star (medium) | `train_arlm_vstar_small.sh`           | -                      | X                            |
| ARM   | Star (hard)   | `train_arlm_vstar_medium.sh`          | -                      | X                            |
| ---   | ---           | ---                                   | ---                    | ---                          |
| IT    | Star (easy)   | `train_it_stochastic_star_small.sh`   | `it-star_easy.pt`      | `it_stochastic_star_small`   |
| IT    | Star (medium) | `train_it_stochastic_vstar_small.sh`  | `it-star_medium.pt`    | `it_stochastic_vstar_small`  |
| IT    | Star (hard)   | `train_it_stochastic_vstar_medium.sh` | `it-star_hard.pt`      | `it_stochastic_vstar_medium` |
| ---   | ---           | ---                                   | ---                    | ---                          |
| XLNet | Star (easy)   | `train_xlnet_star_small.sh`           | `xlnet-star_easy.pt`   | `xlnet_star_small`           |
| XLNet | Star (medium) | `train_xlnet_vstar_small.sh`          | `xlnet-star_medium.pt` | `xlnet_vstar_small`          |
| XLNet | Star (hard)   | `train_xlnet_vstar_medium.sh`         | `xlnet-star_hard.pt`   | `xlnet_vstar_medium`         |
| ---   | ---           | ---                                   | ---                    | ---                          |
| ---   | ---           | ---                                   | ---                    | ---                          |
| ILM   | Zebra         | `train_ilm_tiny2_zebra.sh`            | `ilm-zebra.pt`         | `ilm_tiny2_zebra`            |
| ---   | ---           | ---                                   | ---                    | ---                          |
| MDM   | Zebra         | `train_mdlm_zebra.sh`                 | `mdm-zebra.pt`         | `mdlm_zebra`                 |
| ---   | ---           | ---                                   | ---                    | ---                          |
| ARM   | Zebra         | `train_arlm_zebra.sh`                 | -                      | X                            |
| ---   | ---           | ---                                   | ---                    | ---                          |
| ILM   | LM1B          | `train_ilm_lm1b_multi_node.sh`        | `ilm-lm1b.pt`          | X                            |
| ---   | ---           | ---                                   | ---                    | ---                          |
| MDM   | LM1B          | `train_mdlm_lm1b_multi_node.sh`       | `mdm-lm1b.pt`          | `mdlm_lm1b`                  |
| ---   | ---           | ---                                   | ---                    | ---                          |
| ARM   | LM1B          | `train_arlm_lm1b_multi_node.sh`       | -                      | X                            |
| ---   | ---           | ---                                   | ---                    | ---                          |
| ILM   | Stories       | `train_ilm_stories_multi_node.sh`     | `ilm-stories.pt`       | `ilm_stories`                |
| ---   | ---           | ---                                   | ---                    | ---                          |
| MDM   | Stories       | `train_mdlm_stories_multi_node.sh`    | `mdm-stories.pt`       | `mdlm_stories`               |
| ---   | ---           | ---                                   | ---                    | ---                          |
| ARM   | Stories       | `train_arlm_stories_multi_node.sh`    | -                      | X                            |
| ---   | ---           | ---                                   | ---                    | ---                          |

# Generation

First download all the checkpoints using `python download_weights.py`.


## Unconditional generation

| Model | Dataset | Script                                  |
|-------|---------|-----------------------------------------|
| ILM   | LM1B    | `scripts/generate_ilm_lm1b_timed.sh`    |
| ILM   | Stories | `scripts/generate_ilm_stories_timed.sh` |

## Variable length infill

| Model | Dataset | Script                          |
|-------|---------|---------------------------------|
| ILM   | LM1B    | `scripts/infill_ilm_lm1b.sh`    |
| ILM   | Stories | `scripts/infill_ilm_stories.sh` |



# Evaluation

- **NLL for unconditional generation**

Use `scripts/nll_per_sample.sh` by providing the path to the unconditional generation file.

- **NLL for variable length infill**

Use `scripts/nll_per_sample_infill.sh` by providing the path to the infill generation file.

- **LLM-as-Judge Evaluation for unconditional generation**

Use `src/llm/eval/generate_llm_eval.sh` for LLM-as-Judge evaluation by providing path to judge model, unconditional generation file, judge output file, and the rubrics (coherence | grammaticality | fluency | consistency | spelling_accuracy). We use the judge model [prometheus-eval/prometheus-7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) for our evaluation results.



# Directory Structure

## Inside pcdd/src

1. `commands`: Contains the entry-level scripts like `lightning_main_v2.py` and `lightning_train.py`
2. `models`: Contains complete networks used for generation.
3. `modules`: Contains the building blocks of the networks.
4. `utils`: Contains utility functions.
5. `datamodule`: Contains one file for each dataset.
6. `diffusion`: Contains one file for each type of model. The main object in each file a lightning module.




# Cite

```
@misc{patel2025insertion,
    title={Insertion Language Models: Sequence Generation with Arbitrary-Position Insertions},
    author={Dhruvesh Patel and Aishwarya Sahoo and Avinash Amballa and Tahira Naseem and Tim G. J. Rudner and Andrew McCallum},
    year={2025},
    eprint={2505.05755},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


The code builds on [MDLM](https://github.com/kuleshov-group/mdlm).