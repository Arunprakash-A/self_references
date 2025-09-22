
## Basics of SLURM
Table for command 2
| Hostname        | Status | Resource            |
|-----------------|--------|---------------------|
| node001    | defq*  | gpu:gpu:8(S:0)      |
| node003    | defq*  | gpu:gpu:8(S:0)      |
| node004    | defq*  | gpu:gpu:8(S:0)      |
| node006    | defq*  | gpu:8               |

1. `sinfo` #show available partitions and node status
    - `{PARTITION:defq*, AVAIL:up,  TIMELIMIT:infinite,   NODES:4 ,  STATE:mix, NODELIST:node[001,003-004,006]}`
2. `sinfo -N -h -o "%N %P %G"`   

3. `squeue` #list running and queued jobs
   
4. `scontrol show node` # details info like node-address, node-hostname, gres/gpu-8,
5. `srun ` for running a task 

                   
## Best Practices
- For shared login (i.e, a single login used by multiple people from the project team)
   - Always set up a separate conda environment for each project
   - Use `tmux new -s session_name` with `yourname_task` as a session name to avoid potential conflicts
   - Use `unset HISTFILE` to disable storing the command history of the current bash shell so that it doesn't appear in the command history when other team members log into the account. 
   - Always move the unused `.cache` file to a temporary location to reduce the load on `/home`
   - Store your data and scripts in `scratch (projects)/data/team/user_name/`
- In the SLURM job shell script
    - Avoid usin generic job names such as `#SBATCH --job-name=test`, instead use specif names `#SBATCH --job-name=arun_ao_eval`
    - redirect the logs to the personal folder `#SBATCH --output=project/data/llmteam/arun/logs/slurm-%j.out`
    - Always check `squeue -u llmteam` or filter `squeue -u llmteam | grep arun` before submitting a new job.
    - Estimate the memory requirement to train (eval) the model and use `--gres=gpu:X` only for the number of GPUs you need (don’t lock all 8 unless required)
    - Same goes for time `#SBATCH --time=02:00:00`. Always use model checkpointing so that we can release the resources for other immediate projects if required.
    - Use `scancel <job_id>` instead of `kill`
- Never use `nn.DataParallel` even if you use only 2 GPUs from a node (`--gres=gpu:2`)
- Do a **smoke test** to ensure there is no memory fragmentation in the GPU (so that you can use the available memory optimally)
- For DDP, FSDP, always do a **dry-run** with a small batch of samples and 
   - check proper Gradient Synchronisation
   - Profile GPU and CPU utilisation
   - Check for bottlenecks
- Run the model on the entire dataset only if the dry-run is successful

## CUDA Errors
1. Invalid Device Ordinal (forgot to modify "cuda:7" in a torch.device("cuda:7"), forgetting that you are now running on a single GPU!)    
    - torch.AcceleratorError: CUDA error: invalid **device ordinal**
   CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
   For debugging consider passing CUDA_LAUNCH_BLOCKING=1
   Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
