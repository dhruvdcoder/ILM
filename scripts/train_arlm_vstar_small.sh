#!/bin/bash
#SBATCH -o logs/slurm/%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=30GB                   # server memory requested (per node)
#SBATCH -t 08:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="vram40"
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=1       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=absahoo@umass.edu

# use command: sbatch --job-name=arlm_vstar_small scripts/train_arlm_vstar_small.sh

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
#  debug: true
#  model_type: ???
#  model: ???
#  dataset: ???
#  collation_strategy: ???
#  noise_schedule: ???
# TAGS
job_type=train
debug=false
model_type=arlm
model=rotary_transformer_small_arlm
datamodule=star_arlm
dataset=vstar_small_v2
collation_strategy=default
noise_schedule=dummy
#tokenizer=bert-base-uncased
wandb_job_id=${SLURM_JOB_NAME}_${SLURM_JOB_ID} # wandb job id from command line (use for resuming)


#job_name=\"job_type=${job_type}__model_type=${model_type}__model=${model}__noise_schedule=${noise_schedule}__dataset=${dataset}__collation=${collation_strategy}__debug=${debug}\"
job_name=${wandb_job_id}

# benchmarking
compile=false
precision=32
# print the command
set -x
python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'


srun python -O src/pcdd/commands/lightning_main_v2.py \
    "job_name=${job_name}" \
    "job_type=${job_type}" \
    "model=${model}" \
    "model.force_flash_attn=false" \
    "noise_schedule=${noise_schedule}" \
    "datamodule=${datamodule}" \
    "dataset=${dataset}" \
    "collation_strategy=${collation_strategy}" \
    "model_type=${model_type}" \
    "lightning_module._target_=pcdd.diffusion.arlm.ARLMLightningModuleForStarGraphs" \
    "trainer_strategy=single_device" \
    "trainer.devices=1" \
    "trainer.num_nodes=1" \
    "compile=false" \
    "datamodule.train_dataloader_kwargs.batch_size=${batch_size}" \
    "datamodule.rewrite_manual_cache=false" \
    "optimizer.lr=0.00001" \
    "trainer.max_steps=100000" \
    "trainer.val_check_interval=null" \
    "+trainer.check_val_every_n_epoch=2" \
    "lr_scheduler.num_warmup_steps=500" \
    "predictor.sampling_method=sample_top_k" \
    "+predictor.top=1" \
    "callbacks.checkpoint_monitor.monitor=val/nll" \
    "generative_perplexity.evaluators=null" \
    "generative_perplexity.num_samples=1" \
    "+tags.compile=false" \
    "+tags.precision=32" \
    "+loggers.wandb.resume=allow" \
    "+loggers.wandb.id=${wandb_job_id}"