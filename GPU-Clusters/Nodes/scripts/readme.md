## Basics of SLURM
1. `sinfo` #show available partitions and node status
    - `{PARTITION:defq*, AVAIL:up,  TIMELIMIT:infinite,   NODES:4 ,  STATE:mix, NODELIST:iitmadras[001,003-004,006]}`
2. `squeue` #list running and queued jobs
   - <img width="635" height="128" alt="image" src="https://github.com/user-attachments/assets/c587d7cb-90f1-4510-a4e7-9c7dc8fba2d8" />
3. `scontrol show node` # details info like node-address, node-hostname, gres/gpu-8, 

                   
## Best Practices
- For shared login (i.e, a single login used by multiple people from the project team)
   - Always set up a separate conda environment for each project
   - Use `tmux new -s session_name` with `yourname_task` as a session name to avoid potential conflicts
   - Use `unset HISTFILE` to disable storing the command history of the current bash shell so that it doesn't appear in the command history when other team members log into the account. 
   - Always move the unused `.cache` file to a temporary location to reduce the load on `/home`
   - Store your data and scripts in `scratch (projects)/data/team/user_name/`
- Never use `nn.DataParallel` even if you use only 2 GPUs from a node (`--gres=gpu:2`)
- Do a **smoke test** to ensure there is no memory fragmentation in the GPU (so that you can use the available memory optimally)
- For DDP, FSDP, always do a **dry-run** with a small batch of samples and 
   - check proper Gradient Synchronisation
   - Profile GPU and CPU utilisation
   - Check for bottlenecks
- Run the model on the entire dataset only if the dry-run is successful

## CUDA Errors
1. Set "cuda:7" in a torch.device("cuda:7"), forgetting that you are now running on a single GPU!
    - torch.AcceleratorError: CUDA error: invalid **device ordinal**
   CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
   For debugging consider passing CUDA_LAUNCH_BLOCKING=1
   Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
