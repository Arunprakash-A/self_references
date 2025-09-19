## Useful CLI Commands
- `nvidia-smi` (often, sufficient)
   - Do not use the node directly. Always use SLURM to submit the job
   - You rarely see the node without any process, snap it.
   - <img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/52a745e0-e5fc-4ba3-86ba-09dbc8d7ef4c" />

- [Documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html) for advanced usage
- `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`

## Useful Packages
 - [gpustat](https://pypi.org/project/gpustat/)

## Recommendation
 - Always use PyTorch's `torch.cuda` to get information about the GPU
