#!/bin/bash
#SBATCH -o logs/slurm/nll_dataset_stories.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=20GB                   # server memory requested (per node)
#SBATCH -t 2:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint="vram40,bf16"  # ib for infiniband
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=4       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs


# example usage: sbatch --exclude uri-gpu014 --job-name=mdlm_lm1b -p gpu,gpu-preempt,superpod-a100 scripts/nll.sh
# comment and uncomment relevant sample file paths below and change the name of the job accordingly

export HYDRA_FULL_ERROR=1
# set some NCCL params. See https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html#on-a-multi-node-cluster-set-nccl-parameters
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# See https://pytorch.org/docs/main/torch.compiler_troubleshooting.html for getting right compile logs.
export TORCH_LOGS="recompiles"

experiment=stories_generative_perplexity

python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'
srun python -O src/pcdd/commands/compute_generative_preplexity_for_dataset.py \
    "experiment=${experiment}"
