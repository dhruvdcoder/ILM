#!/bin/bash
#SBATCH -o logs/slurm/nll_per_sample_%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=20GB                   # server memory requested (per node)
#SBATCH -t 4:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint="vram40,bf16"  # ib for infiniband
#SBATCH --ntasks-per-node=1    # should match number of GPUs per node
#SBATCH --cpus-per-task=4       # can be used to set the num_workers
#SBATCH --gres=gpu:1                  # number of GPUs needed per node
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=dhruveshpate@umass.edu


# example usage: sbatch --job-name=lm1b scripts/nll_per_sample.sh
# comment and uncomment relevant sample file paths below and change the name of the job accordingly

export HYDRA_FULL_ERROR=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export TORCH_LOGS="recompiles"


sample_files=(
    "model_weights/ilm-lm1b/unconditional/timed_unconditional_lm1b_max_steps_128_stopping_threshold_0.5_sampling_method_sample_top_p_p_0.9_second_sampling_method_sample_top_k_second_top_1.jsonl"
    "model_weights/ilm-lm1b/unconditional/timed_unconditional_lm1b_max_steps_128_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.9_second_sampling_method_sample_top_k_second_top_1.jsonl"
    "model_weights/ilm-stories/unconditional/unconditional_stories_max_steps_1024_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.8_second_sampling_method_sample_top_k_second_top_1.jsonl"
    "model_weights/ilm-stories/unconditional/unconditional_stories_max_steps_1024_stopping_threshold_0.5_sampling_method_sample_top_p_p_0.8_second_sampling_method_sample_top_k_second_top_1.jsonl"
    "model_weights/ilm-stories/unconditional/unconditional_stories_max_steps_1024_stopping_threshold_0.5_sampling_method_sample_top_p_p_0.2_second_sampling_method_sample_top_k_second_top_1.jsonl"
)


set -x
set -e # exit on the first error
python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'

# Loop over each sample file and run the computation
for samples_file in "${sample_files[@]}"; do
    # Create a unique job name by appending the basename of the sample file (without .jsonl)
    job_name="${wandb_job_id}_$(basename "${samples_file}" .jsonl)"
    echo "Processing ${samples_file} with job name ${job_name}"
    
    srun python -O src/pcdd/commands/add_nll_to_samples.py \
        "job_name=${job_name}" \
        "samples_file=${samples_file}"
done

