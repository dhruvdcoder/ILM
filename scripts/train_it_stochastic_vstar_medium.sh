#!/bin/sh

#SBATCH --constraint          vram80,bf16
#SBATCH --cpus-per-task       5
#SBATCH --gres                gpu:1
#SBATCH --job-name            it_stochastic_vstar_medium
#SBATCH --mail-type           BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user           dhruveshpate@umass.edu
#SBATCH --mem                 20GB
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --open-mode           append
#SBATCH --output              /scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/it_stochastic_vstar_medium/sbatch/2025-07-19_21-56-58/%x.out
#SBATCH --partition           gpu,gpu-preempt,superpod-a100
#SBATCH --requeue             
#SBATCH --time                1-00:00:00

export HYDRA_FULL_ERROR=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export TORCH_LOGS=recompiles
export TQDM_MINITERS=1000
srun python -O src/pcdd/commands/lightning_main_v2.py job_name=it_stochastic_vstar_medium job_type=train experiment=it_stochastic_vstar_medium datamodule.train_dataloader_kwargs.batch_size=64 trainer_strategy=single_device trainer.devices=1 trainer.num_nodes=1 ++trainer.precision=bf16-mixed compile=False +loggers.wandb.resume=allow +loggers.wandb.id=it_stochastic_vstar_medium compile=false trainer.precision=bf16-mixed trainer.num_nodes=1 trainer.devices=1 trainer_strategy=single_device
