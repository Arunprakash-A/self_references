# Simple SLURM Commands Guide

A beginner-friendly guide to essential SLURM commands for running jobs on compute clusters.

## What is SLURM?
SLURM is a job scheduler that manages compute resources on clusters. You request resources (CPUs, GPUs, time) and SLURM finds available nodes to run your jobs.

## Check What's Available

### `sinfo` - See cluster status
```bash
# Show all compute nodes and their status
sinfo

# Show more details about nodes
sinfo -l
```
- **idle** = available to use
- **alloc** = currently being used
- **down** = not working

### `squeue` - See what jobs are running
```bash
# Show all jobs in the queue
squeue

# Show only your jobs
squeue -u $USER
```
- **R** = running
- **PD** = pending (waiting for resources)

## Understanding Resource Options

| Option | What it means | Example |
|--------|---------------|---------|
| `-n` | **Number of tasks** - how many parallel processes | `-n 4` = run 4 processes |
| `-c` | **CPUs per task** - CPU cores for each process | `-c 8` = 8 cores per process |
| `-t` | **Time limit** - how long your job can run | `-t 2:00:00` = 2 hours max |
| `--gres=gpu:X` | **GPUs** - request X graphics cards | `--gres=gpu:2` = 2 GPUs |

**Important:** `-n` vs `-c`
- Use `-n` for parallel jobs (like MPI) - multiple processes working together
- Use `-c` for single jobs that need multiple CPU cores (like threading)
- For most single programs, use `-c` not `-n`

## Interactive Sessions (Development & Testing)

### Basic interactive shell
```bash
# Get an interactive terminal with default resources
srun --pty bash

# Get 4 CPU cores for 1 hour
srun -c 4 -t 1:00:00 --pty bash

# Get 1 GPU and 8 CPU cores for 2 hours
srun -c 8 --gres=gpu:1 -t 2:00:00 --pty bash
```

### Common combinations for `srun --pty bash`
```bash
# Light work: 4 cores, 1 hour
srun -c 4 -t 1:00:00 --pty bash

# Medium work: 8 cores, 1 GPU, 4 hours  
srun -c 8 --gres=gpu:1 -t 4:00:00 --pty bash

# Heavy work: 16 cores, 2 GPUs, 8 hours
srun -c 16 --gres=gpu:2 -t 8:00:00 --pty bash

# Just GPUs: 8 cores, 4 GPUs, 12 hours
srun -c 8 --gres=gpu:4 -t 12:00:00 --pty bash
```

## Running Programs Directly

### `srun` - Run a command immediately
```bash
# Run a program with 4 CPU cores
srun -c 4 -t 1:00:00 python train.py

# Run with GPU
srun -c 8 --gres=gpu:1 -t 2:00:00 python train.py

# Run parallel job (4 processes)
srun -n 4 -t 1:00:00 mpirun ./my_program
```

## Batch Jobs (Submit and Wait)

### `sbatch` - Submit a job script
First create a script file (e.g., `job.sh`):
```bash
#!/bin/bash
#SBATCH -c 8                    # 8 CPU cores
#SBATCH --gres=gpu:1            # 1 GPU
#SBATCH -t 4:00:00              # 4 hours max
#SBATCH -J my_job_name          # Job name
#SBATCH -o output_%j.log        # Output file (%j = job ID)

# Your commands here
python train.py
```

Then submit it:
```bash
sbatch job.sh
```

## Managing Jobs

### Cancel jobs
```bash
# Cancel a specific job (replace 12345 with actual job ID)
scancel 12345

# Cancel all your jobs
scancel -u $USER
```

### Check job details
```bash
# See detailed info about a job
scontrol show job 12345
```

## Quick Reference

### Most common commands:
```bash
# Check what's available
sinfo
squeue

# Get interactive session with GPU
srun -c 8 --gres=gpu:1 -t 4:00:00 --pty bash

# Run a program directly
srun -c 4 --gres=gpu:1 -t 2:00:00 python my_script.py

# Submit a job script
sbatch my_job.sh

# Check your jobs
squeue -u $USER

# Cancel a job
scancel JOB_ID
```

## Tips for Beginners
- Start with small time limits (`-t 1:00:00`) while learning
- Use `srun --pty bash` to test your setup before submitting big jobs  
- Always check `squeue -u $USER` to monitor your jobs
- Use `sinfo` to see if GPUs are available before requesting them
- Job IDs are shown when you submit jobs - write them down!
- If your job is "PD" (pending) for a long time, try requesting fewer resources

**Credit:** Received from Ashwin Shankar, AI4Bharat, IIT Madras 
