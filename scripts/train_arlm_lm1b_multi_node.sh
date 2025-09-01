#!/bin/bash
#SBATCH -o logs/slurm/%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 4                          # Total number of nodes requested
#SBATCH --mem=20GB                   # server memory requested (per node)
#SBATCH -t 4-00:00:00                  # Time limit (hh:mm:ss)
#SBATCH --qos=long
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint="vram80,bf16,ib"  # ib for infiniband
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=5       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=absahoo@umass.edu
#SBATCH -A pi_mccallum_umass_edu

export HYDRA_FULL_ERROR=1
# set some NCCL params. See https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html#on-a-multi-node-cluster-set-nccl-parameters
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export TQDM_MINITERS=1000

# See https://pytorch.org/docs/main/torch.compiler_troubleshooting.html for getting right compile logs.
export TORCH_LOGS="recompiles"

# things that depend on GPU size, sequence length, and model size. We set them here.
# per device batch size -> datamodule.train_dataloader_kwargs.batch_size
# generative perplexity batch size -> generative_perplexity.evaluators.[evaluator_name].batch_size
batch_size=64
gen_batch_size=8



experiment=arlm_lm1b
job_type=train
debug=false
model_type=arlm
model=rotary_transformer_small_arlm
datamodule=arlm_lm1b
dataset=lm1b_v2
collation_strategy=default
noise_schedule=dummy
tokenizer=bert-base-uncased
wandb_job_id=${SLURM_JOB_NAME}_29963094 # wandb job id from command line (use for resuming)


job_name=${wandb_job_id}
job_type=train

# benchmarking
compile=false
precision=bf16-mixed
#gpu=rtx8000
# print the command
set -x
python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'


srun python -O src/pcdd/commands/lightning_main_v2.py \
    "job_name=${job_name}" \
    "job_type=${job_type}" \
    "experiment=${experiment}" \
    "trainer_strategy=ddp_multinode" \
    "trainer.devices=1" \
    "trainer.num_nodes=4" \
    "++trainer.precision=${precision}" \
    "compile=${compile}" \
    "datamodule.train_dataloader_kwargs.batch_size=${batch_size}" \
    "generative_perplexity.evaluators=null" \
    "datamodule.rewrite_manual_cache=false" \
    "optimizer.lr=0.0001" \
    "callbacks.checkpoint_monitor.monitor=val/nll" \
    "hydra.job.env_set.TQDM_MINITERS=${TQDM_MINITERS}" \
    "+tags.compile=${compile}" \
    "+tags.precision=${precision}" \
    "+loggers.wandb.resume=allow" \
    "+loggers.wandb.id=${wandb_job_id}" \
    "+loggers.wandb.notes='Regular ARLM for LM1B.'"