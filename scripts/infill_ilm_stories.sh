#!/bin/bash
#SBATCH -o logs/slurm/%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=10GB                   # server memory requested (per node)
#SBATCH -t 02:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint="vram40,bf16"  # ib for infiniband
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=4       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs

export HYDRA_FULL_ERROR=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export TORCH_LOGS="recompiles"


infill_dataset=tiny_stories
experiment=ilm_stories_infill
ckpt_path=model_weights/ilm-stories/ilm-stories.pt
infill_dataset=tiny_stories
infill_percent=1 # 1 sentence version
predictor_max_steps=1024
stopping_threshold=0.9
sampling_method=sample_top_p
p=0.8
top=1
second_sampling_method=sample_top_k
second_top=1
second_p=0.1 # not used

output_dir=model_weights/ilm-stories/infill-${infill_percent}

output_file_name=infill_max_steps_${predictor_max_steps}_stopping_threshold_${stopping_threshold}_sampling_method_${sampling_method}_top_${top}_p_${p}_second_sampling_method_${second_sampling_method}_second_top_${second_top}_second_p_${second_p}.jsonl

wandb_job_id=${1:-${SLURM_JOB_NAME}} # wandb job id from command line (use for resuming)


job_name=${wandb_job_id}


set -x
python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'


srun python -u -O src/pcdd/commands/lightning_main_v2.py \
    "job_name=${job_name}" \
    "experiment=${experiment}" \
    "generation.datamodule.infill_dataset_name=${infill_dataset}" \
    "generation.datamodule.infill_percent=${infill_percent}" \
    "++generation.model_only_checkpoint_path=${ckpt_path}" \
    "++generation.ckpt_path=null" \
    "generation.output_dir=${output_dir}" \
    "generation.output_file_name=${output_file_name}" \
    "++predictor.max_steps=${predictor_max_steps}" \
    "++predictor.stopping_threshold=${stopping_threshold}" \
    "++predictor.sampling_method=${sampling_method}" \
    "++predictor.top=${top}" \
    "++predictor.p=${p}" \
    "++predictor.second_sampling_method=${second_sampling_method}" \
    "++predictor.second_top=${second_top}" \
    "++predictor.second_p=${second_p}" \
    "++predictor.force_predict_first_step=true"