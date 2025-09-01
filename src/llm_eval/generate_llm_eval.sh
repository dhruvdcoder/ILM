#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint=[bf16]
#SBATCH -A pi_hongyu_umass_edu
# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

input_file="$1"
model_repo="$2"
output_file="$3"
rubrics_name="$4"

python generate_llm_eval.py --model_repo $model_repo --input_file $input_file --output_file $output_file --rubrics_name $rubrics_name