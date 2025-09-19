## Best Practices
- Never use `nn.DataParallel` even if your node has only 2 GPUs
- Do a **smoke test** to ensure there is no memory fragmentation in GPU (so that you can use the available memory optimally)
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
