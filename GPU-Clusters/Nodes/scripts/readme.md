
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
## Submitting and cancelling Job
* `sbatch train_job.slurm`
  - prints "submitted job with jobid 21454"
* To see the details of the job  `scontrol show job 21454`
    ```
     JobId=21454 JobName=arun_test_ao
       UserId=llmteam(1005) GroupId=llmteam(1005) MCS_label=N/A
       Priority=4294901671 Nice=0 Account=(null) QOS=normal
       JobState=PENDING Reason=Resources Dependency=(null)
       Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
       RunTime=00:00:00 TimeLimit=01:00:00 TimeMin=N/A
       SubmitTime=2025-09-28T11:34:30 EligibleTime=2025-09-28T11:34:30
       AccrueTime=2025-09-28T11:34:30
       StartTime=2026-09-22T13:43:14 EndTime=2026-09-22T14:43:14 Deadline=N/A
       SuspendTime=None SecsPreSuspend=0 LastSchedEval=2025-09-28T11:44:50 Scheduler=Main
       Partition=defq AllocNode:Sid=iitmadras-login:4116585
       ReqNodeList=iitmadras001 ExcNodeList=(null)
       NodeList=
       NumNodes=1-1 NumCPUs=8 NumTasks=1 CPUs/Task=8 ReqB:S:C:T=0:0:*:*
       ReqTRES=cpu=8,mem=80G,node=1,billing=8,gres/gpu=4
       AllocTRES=(null)
       Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
       MinCPUsNode=8 MinMemoryNode=80G MinTmpDiskNode=0
       Features=(null) DelayBoot=00:00:00
       OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
       Command=/projects/data/llmteam/arun/train_job.slurm
       WorkDir=/projects/data/llmteam/arun
       StdErr=/projects/data/llmteam/arun/logs/arun_test_ao_21454_%N_%t.err
       StdIn=/dev/null
       StdOut=/projects/data/llmteam/arun/logs/arun_test_ao_21454_%N_%t.log
       Power=
       TresPerNode=gres:gpu:4
     ```
  * Cancelling the job `scancel 21454`         
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
    - Use `scancel <job_id>` instead of `kill` (Note down the Jobid for future reference)
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
