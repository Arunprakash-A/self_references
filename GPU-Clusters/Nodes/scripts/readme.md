# Best Practices
- Never use `nn.DataParallel` even if your node has only 2 GPUs
- For DDP, FSDP, always to **dry-run** with a small batch of samples and 
   - check proper Gradient Synchronisation
   - Profile GPU and CPU utilisation
   - Check for bottlenecks
- Run the model on the entire dataset only if the dry-run is successful
