#!/bin/bash
#SBATCH -o logs/slurm/%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=30GB                   # server memory requested (per node)
#SBATCH -t 12:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu
#SBATCH --constraint="vram40,bf16"
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=4       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=dhruveshpate@umass.edu

# use command: sbatch --job-name=ilm_stopping_vstar_medium scripts/train_ilm_stopping_vstar_medium.sh

#echo "cd to project root"
#cd .. || exit 1

#TODO (sbatch): Add documentation
# See https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/main/scripts/train_lm1b.sh

export HYDRA_FULL_ERROR=1

# things that depend on GPU size, sequence length, and model size. We set them here.
# per device batch size -> datamodule.train_dataloader_kwargs.batch_size
# generative perplexity batch size -> generative_perplexity.evaluators.[evaluator_name].batch_size
batch_size=64
gen_batch_size=8


#tags:
#  debug: false
#  model_type: ???
#  model: ???
#  dataset: ???
#  collation_strategy: ???
#  noise_schedule: ???
# TAGS
job_type=train
debug=false
model_type=ilm_stopping
model=rotary_transformer_small_ilm_stopping
datamodule=star_ilm_stopping
dataset=star_small_v2
collation_strategy=default
noise_schedule=dummy
#tokenizer=bert-base-uncased
wandb_job_id=${1:-${SLURM_JOB_NAME}} # wandb job id from command line (use for resuming)


#job_name=\"job_type=${job_type}__model_type=${model_type}__model=${model}__noise_schedule=${noise_schedule}__dataset=${dataset}__collation=${collation_strategy}__debug=${debug}\"
job_name=${wandb_job_id}

# benchmarking
compile=false
precision=bf16-mixed
# print the command
set -x
python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'


srun python -O src/pcdd/commands/lightning_main_v2.py \
    "job_name=${job_name}" \
    "job_type=${job_type}" \
    "model=${model}" \
    "++model.final_layer_without_normalization=true" \
    "model.force_flash_attn=false" \
    "noise_schedule=${noise_schedule}" \
    "datamodule=${datamodule}" \
    "dataset=${dataset}" \
    "collation_strategy=${collation_strategy}" \
    "model_type=${model_type}" \
    "lightning_module._target_=pcdd.diffusion.ilm_v2.ILMWithStoppingClassificationLightningModuleForStarGraphs" \
    "trainer_strategy=single_device" \
    "trainer.devices=1" \
    "trainer.num_nodes=1" \
    "compile=${compile}" \
    "++trainer.precision=${precision}" \
    "datamodule.train_dataloader_kwargs.batch_size=${batch_size}" \
    "datamodule.rewrite_manual_cache=false" \
    "optimizer.lr=0.0001" \
    "trainer.max_steps=50000" \
    "trainer.val_check_interval=null" \
    "+trainer.check_val_every_n_epoch=2" \
    "lr_scheduler.name=constant" \
    "lr_scheduler.num_warmup_steps=500" \
    "predictor.sampling_method=sample_top_k" \
    "predictor.top=1" \
    "+predictor.second_sampling_method=sample_top_k" \
    "+predictor.second_top=1" \
    "callbacks.checkpoint_monitor.monitor=val/accumulated_loss" \
    "generative_perplexity.evaluators=null" \
    "+tags.compile=${compile}" \
    "+tags.precision=${precision}" \
    "+loggers.wandb.resume=allow" \
    "+loggers.wandb.id=${wandb_job_id}" \
    "+loggers.wandb.notes='Uses masked CE loss that masks out padding and dropped tokens  (not prefix/constraint). Also uses final layer without normalization.'"

