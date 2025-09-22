# Documenting learnings, ideas, interesting-insightful-helpful papers, blogs and scripts (for myself and the curious minds like you)

## Cool Ideas
1. [MegaKernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
    - Comment: If we can fuse a single attention module, why not fuse the entire model to get low latency?
## Mixed Precision training
1. Users **should not** manually cast their model or data to bf16, always safe to first use `amp.autocast`
2. [Best practices guide](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

## Papers
1. [Emergent Abilities of Large Language Models](https://arxiv.org/pdf/2206.07682)
    - Comment: What really happens when we scale the model? Some abilities emerges. In-context learning ( great sample efficiency) is a commonly known ability.
    - Going through the paper gives us even more deeper insights
       - In 2022, many people noticed the advantage of scaling the model parameters, however, one could have also noticed that scaling the FLOPS (compute) also help the model. 
       - This is now called **test-time scaling** (Scaling the FLOPS)
    - The quote "Emergence is when _quantitative_ changes in a system result in _qualitative_ changes in behaviour" cited in the paper deeply resonates with me
2. [TorchAO: PyTorch-Native Training-to-Serving Model Optimization](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf)
    - Comment: The end-to-end optimization of the LLM pipeline (training-finetuning-serving) was fragmented. This framework exactly addresses this major problem.
    - From the paper:
       - For instance, a researcher may pre-train their model using mixed FP8/BF16 precision support in Transformer Engine (NVIDIA, 2025), load the pre-trained model
into Unsloth (Han & Han, 2025) or Axolotl (Axolotl, 2025) for further fine-tuning, perform quantization using bitsandbytes (Dettmers et al., 2022) before finally serving the model
using llama.cpp (GGML, 2025). In each step, the user may need to manually convert the model format (e.g. from HuggingFace’s safetensors to GGUF in llama.cpp), and the
quantization schemes may diverge from the ones used in previous steps with subtle discrepancies
## Best practices
1. **On Training:**
    - Save checkpoints with all intermediate informations, including the current `epoch`, so we can stop and resume training without breaking epoch numbers.
    - This avoids many subtle errors while generating final figures.
    - ```python
      save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), #
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "args": vars(args),
            }, best_ckpt)
      ```
 2. What if the model underperforms?
     - Keep the batch size  small during the initial iterations and increase the size as the iterations increase (adjust warm-up lr accordingly)
     - Use hooks to debug the gradients, activation distribution. If you are using DDP, try dynamic tanh activation by removing batch/layer normalisation and see if the model performance improves
     - In LLMs, you may also vary the beam size while dealing with reasoning models. For example, page 6 in [the paper](https://arxiv.org/pdf/2103.03874)
      
## Some subtle mistakes 
1. If you are on a shared cluster (servers), you should be careful and ensure every step you follow is correct. Do not ASSUME anything
    - For example, I set up a conda environment and installed all required packages.
    - I then used `ipython` for initial experimentation (that is the only option I have due to security concerns).
    - Everything was running smoothly, so I didn't doubt anything.
    - One day, the code threw an error. I spent half a day going through the documentation to fix it.
    - Nothing worked!
    - Finally, I realiszd, I didn't install `ipython` at all in `my environment`, therefore, it used the `ipython` from the base environment (sic!)
    - The base environment has a different PyTorch version than I intended!
2. Sometimes the issue is not related to your code. However, you encounter errors related to the CUDA driver (especially on shared clusters).
    - Attempt clearing the cache directory (`.cache/torch/kernels`) as discussed [here](https://discuss.pytorch.org/t/torch-prod-produces-runtimeerror-cuda-driver-error-invalid-argument/179054/29)
