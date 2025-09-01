#!/bin/bash
#SBATCH -o logs/slurm/nll_%x.out          # output file (%j expands to jobID, %x expands to job_name) # TODO: For resuming, use %x only.
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=10GB                   # server memory requested (per node)
#SBATCH -t 1:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu
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

#python src/pcdd/commands/lightning_main_v2.py experiment=mdlm_stories_generate generation.ckpt_path=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/mdlm_stories/checkpoints/11-50000.ckpt generation.output_dir=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/mdlm_stories/predictions/11-50000 generation.output_file_name=unconditional_stories_max_steps_512 predictor.max_steps=512
#samples_file=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/mdlm_stories/predictions/11-50000/unconditional_stories_max_steps_512.jsonl
#samples_file=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/mdlm_stories/predictions/11-50000/unconditional_stories_max_steps_128.jsonl
#samples_file=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/mdlm_stories/predictions/11-50000/unconditional_stories_max_steps_2.jsonl
# ilm

# Get wandb job id from the command line (or fall back)
wandb_job_id=${1:-nll_${SLURM_JOB_NAME}}

# Create an array of sample files
# ilm_stories
#sample_files=(
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_stories_v2/predictions/18-72000/unconditional_stories_max_steps_1024_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.1_second_sampling_method_sample_top_k_second_top_1.jsonl"
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_stories_v2/predictions/18-72000/unconditional_stories_max_steps_1024_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.2_second_sampling_method_sample_top_k_second_top_1.jsonl"
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_stories_v2/predictions/18-72000/unconditional_stories_max_steps_1024_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.5_second_sampling_method_sample_top_k_second_top_1.jsonl"
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_stories_v2/predictions/18-72000/unconditional_stories_max_steps_1024_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.9_second_sampling_method_sample_top_k_second_top_1.jsonl"
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_stories_v2/predictions/18-72000/unconditional_stories_max_steps_1024_stopping_threshold_0.9_sampling_method_sample.jsonl"
#)

# ilm_lm1b
#sample_files=(
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_lm1b_multinode_v5/predictions/16-1000000/unconditional_stories_max_steps_128_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.1_second_sampling_method_sample_top_k_second_top_1.jsonl"
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_lm1b_multinode_v5/predictions/16-1000000/unconditional_stories_max_steps_128_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.2_second_sampling_method_sample_top_k_second_top_1.jsonl"
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_lm1b_multinode_v5/predictions/16-1000000/unconditional_stories_max_steps_128_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.5_second_sampling_method_sample_top_k_second_top_1.jsonl"
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_lm1b_multinode_v5/predictions/16-1000000/unconditional_stories_max_steps_128_stopping_threshold_0.9_sampling_method_sample_top_p_p_0.9_second_sampling_method_sample_top_k_second_top_1.jsonl"
#"/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/pcdd/logs/ilm_lm1b_multinode_v5/predictions/16-1000000/unconditional_stories_max_steps_128_stopping_threshold_0.9_sampling_method_sample.jsonl"
#)
# mdlm_lm1b
sample_files=(
"/scratch3/workspace/dhruveshpate_umass_edu-pcdd-p/pcdd-p/logs/lm1b_absorbing_predict_pad_legacy/predictions/uncoditional_lm1b_max_steps_128.jsonl"
)
set -x
python -c 'import torch; print("num_gpus: ", torch.cuda.device_count())'

# Loop over each sample file and run the computation
for samples_file in "${sample_files[@]}"; do
    # Create a unique job name by appending the basename of the sample file (without .jsonl)
    job_name="${wandb_job_id}_$(basename "${samples_file}" .jsonl)"
    echo "Processing ${samples_file} with job name ${job_name}"
    
    srun python -O src/pcdd/commands/compute_generative_preplexity.py \
        "job_name=${job_name}" \
        "samples_file=${samples_file}"
done

