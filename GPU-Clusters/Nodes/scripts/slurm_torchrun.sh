#!/bin/bash
#SBATCH --job-name=arun_test_ao
#SBATCH --output=/projects/data/llmteam/arun/logs/%x_%j_%N_%t.log
#SBATCH --error=/projects/data/llmteam/arun/logs/%x_%j_%N_%t.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --nodelist=iitmadras001

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch_ao

# Run the training using torchrun for DDP
torchrun --nproc_per_node=1 vit_h_14.py --dataset dataset --batch_size 2 --epochs 1
