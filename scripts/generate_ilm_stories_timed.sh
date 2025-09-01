#!/bin/bash
#SBATCH -o logs/slurm/%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=10GB                   # server memory requested (per node)
#SBATCH -t 2:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,superpod-a100,gpu-preempt
#SBATCH --constraint="a100-80g"  # ib for infiniband
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=4       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=dhruveshpate@umass.edu

export HYDRA_FULL_ERROR=1
# set some NCCL params. See https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html#on-a-multi-node-cluster-set-nccl-parameters
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# See https://pytorch.org/docs/main/torch.compiler_troubleshooting.html for getting right compile logs.
export TORCH_LOGS="recompiles"

#python src/pcdd/commands/lightning_main_v2.py experiment=mdlm_stories_generate generation.ckpt_path=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/mdlm_stories/checkpoints/11-50000.ckpt generation.output_dir=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/mdlm_stories/predictions/11-50000 generation.output_file_name=unconditional_stories_max_steps_512 predictor.max_steps=512

experiment=ilm_stories_generate_timed
ckpt_path=model_weights/ilm-stories/ilm-stories.pt
output_dir=model_weights/ilm-stories/unconditional



# 2: vanilla
predictor_max_steps=1024
#stopping_threshold=0.5
stopping_threshold=${1}
sampling_method=sample_top_p
second_sampling_method=sample_top_k
#p=0.2
p=${2}
second_top=1
#output_file_name=timed_unconditional_stories_max_steps_${predictor_max_steps}_stopping_threshold_${stopping_threshold}_sampling_method_${sampling_method}.jsonl
output_file_name=unconditional_stories_max_steps_${predictor_max_steps}_stopping_threshold_${stopping_threshold}_sampling_method_${sampling_method}_p_${p}_second_sampling_method_${second_sampling_method}_second_top_${second_top}.jsonl
wandb_job_id=${SLURM_JOB_NAME} # wandb job id from command line (use for resuming)


job_name=${wandb_job_id}


set -x
python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'


srun python -O -u src/pcdd/commands/lightning_main_v2.py \
    "job_name=${job_name}" \
    "experiment=${experiment}" \
    "generation.ckpt_path=null" \
    "++generation.model_only_checkpoint_path=${ckpt_path}" \
    "generation.output_dir=${output_dir}" \
    "generation.output_file_name=${output_file_name}" \
    "++predictor.max_steps=${predictor_max_steps}" \
    "++predictor.stopping_threshold=${stopping_threshold}" \
    "++predictor.sampling_method=${sampling_method}" \
    "++predictor.p=${p}" \
    "++predictor.second_sampling_method=${second_sampling_method}" \
    "++predictor.second_top=${second_top}"

