#!/bin/bash
#SBATCH --job-name=ddp-torchrun-job     # Job name
#SBATCH --nodes=3                       # Number of nodes
#SBATCH --ntasks-per-node=8             # One task per GPU
#SBATCH --gpus-per-node=8               # GPUs per node
#SBATCH --time=12:00:00                 # Walltime (HH:MM:SS)
#SBATCH --mem=0                         # All memory on each node
#SBATCH --exclusive                     # Full node usage
#SBATCH --output=slurm-%j.out           # Output file

# Load modules (adjust per cluster setup)
module load cuda/12.1
module load python/3.10

# Activate your environment if needed
# source ~/envs/myenv/bin/activate

# -----------------------------
# Torchrun configuration
# -----------------------------

# Number of GPUs per node
GPUS_PER_NODE=8
# Total number of nodes
NNODES=$SLURM_NNODES
# Rank of this node
NODE_RANK=$SLURM_NODEID
# Get first node hostname (used as rendezvous address)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Master port (pick a free one)
MASTER_PORT=29500

echo "Nodes allocated: $SLURM_JOB_NODELIST"
echo "Master node: $MASTER_ADDR"
echo "Node rank: $NODE_RANK / $NNODES"

# Launch distributed training
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py --epochs 100 --batch-size 256
