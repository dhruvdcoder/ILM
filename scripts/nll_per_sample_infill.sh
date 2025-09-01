#!/bin/bash
#SBATCH -o logs/slurm/nll_per_sample_%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=10GB                   # server memory requested (per node)
#SBATCH -t 08:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,superpod-a100
#SBATCH --constraint="vram40,bf16"  # ib for infiniband
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=4       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=dhruveshpate@umass.edu



export HYDRA_FULL_ERROR=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

export TORCH_LOGS="recompiles"

# Get wandb job id from the command line (or fall back)
wandb_job_id=${1:-nll_per_sample_${SLURM_JOB_NAME}}


sample_files=(
    "model_weights/ilm-lm1b/infill-multi/infill_max_steps_128_stopping_threshold_0.8_sampling_method_sample_top_p_p_0.2_top_1_second_sampling_method_sample_top_k_second_p_0.8_second_top_1.jsonl"
    "model_weights/ilm-lm1b/infill-single/infill_max_steps_128_stopping_threshold_0.8_sampling_method_sample_top_p_p_0.2_top_1_second_sampling_method_sample_top_k_second_p_0.8_second_top_1.jsonl"
    "model_weights/ilm-stories/infill-1/infill_max_steps_1024_stopping_threshold_0.5_sampling_method_sample_top_p_top_1_p_0.2_second_sampling_method_sample_top_k_second_top_1_second_p_0.1.jsonl"
    "model_weights/ilm-stories/infill-1/infill_max_steps_1024_stopping_threshold_0.5_sampling_method_sample_top_p_top_1_p_0.8_second_sampling_method_sample_top_k_second_top_1_second_p_0.1.jsonl"
)


set -x
#set -e # exit on the first error
python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'

# Loop over each sample file and run the computation
for samples_file in "${sample_files[@]}"; do
    # Create a unique job name by appending the basename of the sample file (without .jsonl)
    job_name="${wandb_job_id}_$(basename "${samples_file}" .jsonl)"
    echo "Processing ${samples_file} with job name ${job_name}"
    
    srun python -O src/pcdd/commands/add_nll_to_samples_infill.py \
        "job_name=${job_name}" \
        "samples_file=${samples_file}"
done

