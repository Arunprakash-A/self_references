## Useful CLI Commands
- `nvidia-smi` (often, sufficient)
- [Documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html) for advanced usage
- `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`

## Useful Packages
 - [gpustat](https://pypi.org/project/gpustat/)

## Recommendation
 - Always use PyTorch's `torch.cuda` to get information about the GPU
