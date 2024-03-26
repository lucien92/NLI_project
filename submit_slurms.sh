#!/bin/bash

#SBATCH --job-name=finetuning_everything
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=40:00:00
#SBATCH --output=project/logslurms/slurm-%A_%a.out
#SBATCH --error=project/logslurms/slurm-%A_%a.err

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

# echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "Activating local environment"

source /usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_32/NLI_project/venv/bin/activate 

pip list

python3 train.py