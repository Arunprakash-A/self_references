# Best Practices
- Never use `nn.DataParallel` even if your node has only 2 GPUs
- Do a **smoke test** to ensure there is no memory fragmentation in GPU (so that you can use the available memory optimally)
- For DDP, FSDP, always do a **dry-run** with a small batch of samples and 
   - check proper Gradient Synchronisation
   - Profile GPU and CPU utilisation
   - Check for bottlenecks
- Run the model on the entire dataset only if the dry-run is successful
