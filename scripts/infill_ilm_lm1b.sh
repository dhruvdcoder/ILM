#!/bin/bash
#SBATCH -o logs/slurm/%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=10GB                   # server memory requested (per node)
#SBATCH -t 2:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --constraint="vram40,bf16"  # ib for infiniband
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=4       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs

export HYDRA_FULL_ERROR=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

export TORCH_LOGS="recompiles"




experiment=ilm_lm1b_infill
output_dir=model_weights/ilm-lm1b/infill-${infill_type}
ckpt_path=model_weights/ilm-lm1b/ilm-lm1b.pt

# multi infill predictor options
infill_type=multi
max_steps=128
stopping_threshold=0.8
sampling_method=sample_top_p
p=0.2
top=1
second_sampling_method=sample_top_k
second_top=1
second_p=0.8


wandb_job_id=${1:-${SLURM_JOB_NAME}} # wandb job id from command line (use for resuming)
job_name=${wandb_job_id}_${infill_type}


srun python -u -O src/pcdd/commands/lightning_main_v2.py \
    "job_name=${job_name}" \
    "experiment=${experiment}" \
    "generation.datamodule.infill_type=${infill_type}" \
    "++generation.model_only_checkpoint_path=${ckpt_path}" \
    "++generation.ckpt_path=null" \
    "generation.output_dir=${output_dir}" \
    "++predictor.max_steps=${max_steps}" \
    "++predictor.stopping_threshold=${stopping_threshold}" \
    "++predictor.sampling_method=${sampling_method}" \
    "++predictor.p=${p}" \
    "++predictor.top=${top}" \
    "++predictor.second_sampling_method=${second_sampling_method}" \
    "++predictor.second_top=${second_top}" \
    "++predictor.second_p=${second_p}"




# single infill predictor options
infill_type=single
max_steps=128
stopping_threshold=0.8
sampling_method=sample_top_p
p=0.2
top=1
second_sampling_method=sample_top_k
second_top=1
second_p=0.8


wandb_job_id=${1:-${SLURM_JOB_NAME}} # wandb job id from command line (use for resuming)
job_name=${wandb_job_id}_${infill_type}


srun python -u -O src/pcdd/commands/lightning_main_v2.py \
    "job_name=${job_name}" \
    "experiment=${experiment}" \
    "generation.datamodule.infill_type=${infill_type}" \
    "++generation.model_only_checkpoint_path=${ckpt_path}" \
    "++generation.ckpt_path=null" \
    "generation.output_dir=${output_dir}" \
    "++predictor.max_steps=${max_steps}" \
    "++predictor.stopping_threshold=${stopping_threshold}" \
    "++predictor.sampling_method=${sampling_method}" \
    "++predictor.p=${p}" \
    "++predictor.top=${top}" \
    "++predictor.second_sampling_method=${second_sampling_method}" \
    "++predictor.second_top=${second_top}" \
    "++predictor.second_p=${second_p}"