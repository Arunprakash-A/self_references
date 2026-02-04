# Documenting learnings, ideas, interesting-insightful-helpful papers, blogs and scripts (for myself and the curious minds like you)
## Books/Chapters 
1. [PATTERNS, PREDICTIONS, AND ACTIONS A story about machine learning](https://mlstory.org/pdf/patterns.pdf)
   - Surprised to learn the story behind the modern concept sitting at the centre of RL:'sum of expected discounted rewards`
      - Excerpt from the book: "Halley recognized that his table could be used to estimate the odds that a person of a certain age would die within the next year. Based on this
observation, he described a formula for pricing an annuity that, expressed in modern language, computes the sum of expected discounted payouts over the course of a person’s life starting from their current age".
2. [The Alignment Problem](https://brianchristian.org/the-alignment-problem/)
   - It is a must-read (of course, it is an AI book)
   - Contains a lot of real-life incidents like sharing the moments while AlexNet was being cooked, ethical aspects (and their importance), and limitations of models
3. [The Intelligent Machinery by Turing](https://dn790006.ca.archive.org/0/items/turing1948/turing1948_text.pdf)
   - Just sharing the pieces of information I loved from the book
       - The analogy with the human brain is used as a **guiding principle**
       - "We simply do not know of any creation which goes on creating itself in variety when the creator has withdrawn from it" <-- take on the definition of God :-)
       - An unwillingness to admit the possibility that mankind can have any rivals in intellectual power. This occurs as much amongst intellectual people as amongst others: they have more to lose
       - Paper Machine:
          - It is possible to produce the effect of a computing machine by writing down a set of rules of procedure and asking a man to carry them out. Such a combination of a man with written instructions will               be called a paper Machine’. A man provided with paper, pencil, and rubber, and subject to strict discipline, is in effect a universal machine.
      - Although the operations which can be performed by lcms include every ruleof-thumb process, the number of steps involved tends to be enormous. This is mainly due to the arrangement of the memory along the         tape
          - Two facts which need to be used together may be stored very far apart on the tape. There is also rather little encouragement, when dealing with these machines, to condense the stored expressions at                all <-- Insight into the problem
      - The sections: Man as a Machine and Education of Machinery are really good.
4. [The Unreasonable Effectiveness of Mathematics in the Natural Sciences](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/wigner.pdf)
      - Introductory Joke: Why does pi appear everywhere despite its original invention as a ratio of circumference to the diameter?
         - Appearance of pi in Gaussian distribution (something to do with the population :-))
      - How come mathematics, as an independent study, finds its application in science and engineering?
      - All mathematical concepts are learned from our experience of the real world, so how come imaginary numbers were invented?
      - How do we define the laws of nature? regularity  in the complexity? order in chaos?
      -  .... a miracle that in spite of the baffling complexity of the world, certain regularities in the events could be discovered
      -  Laws are invariant across the universe
      -  ... regularity which we are discussing is **independent** of so many conditions which could have an effect on it


   

## Cool Ideas
1. [MegaKernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
    - Comment: If we can fuse a single attention module, why not fuse the entire model to get low latency?
## Mixed Precision training
1. Users **should not** manually cast their model or data to bf16, always safe to first use `amp.autocast`
2. [Best practices guide](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

## Reinforcement Learning
1. [Understanding Reinforcement Learning for Model Training, and future directions with GRAPE](https://arxiv.org/pdf/2509.04501)
   - Comment: Good read for anyone who has a good knowledge of LLM and wants to learn the role of RL algorithms
2. [Reinforcement Learning - An overview, by Kevin P.Murphy](https://arxiv.org/pdf/2412.05265)



## Papers
1. [Emergent Abilities of Large Language Models](https://arxiv.org/pdf/2206.07682)
    - **Comment:** What really happens when we scale the model? Some abilities emerges. In-context learning ( great sample efficiency) is a commonly known ability.
    - Going through the paper gives us even more deeper insights
       - In 2022, many people noticed the advantage of scaling the model parameters, however, one could have also noticed that scaling the FLOPS (compute) also help the model. 
       - This is now called **test-time scaling** (Scaling the FLOPS)
    - The quote "Emergence is when _quantitative_ changes in a system result in _qualitative_ changes in behaviour" cited in the paper deeply resonates with me
2. [Training with Verifiers](https://arxiv.org/pdf/2110.14168)
      - **Comment:**, IMO, First paper that demonstrates (taking ideas from earlier research), test-time compute is an alternative path to explore (the paper shows it for math reasoning)
      - 6B model with increased compute (i.e., generating 100 solutions and verifying them) performs better than fine-tuning 175B model.
      - However, it also demonstrates that naively increasing the compute (say, increasing the completions to 200) won't increase the performance. There are some sweet spots
2. [TorchAO: PyTorch-Native Training-to-Serving Model Optimization](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf)
    - **Comment:** The end-to-end optimization of the LLM pipeline (training-finetuning-serving) was fragmented. This framework exactly addresses this major problem.
    - From the paper:
       - For instance, a researcher may pre-train their model using mixed FP8/BF16 precision support in Transformer Engine (NVIDIA, 2025), load the pre-trained model
into Unsloth (Han & Han, 2025) or Axolotl (Axolotl, 2025) for further fine-tuning, perform quantization using bitsandbytes (Dettmers et al., 2022) before finally serving the model
using llama.cpp (GGML, 2025). In each step, the user may need to manually convert the model format (e.g. from HuggingFace’s safetensors to GGUF in llama.cpp), and the
quantization schemes may diverge from the ones used in previous steps with subtle discrepancies

3. [Solving Math Word Problems](https://aclanthology.org/D17-1088.pdf)
    - **Comment:** Reasoning models are not just for solving complex mathematical equations; it is a foundational skill that enables the adoption of LLMs in many areas that require strong reasoning to make decisions.
    - Math word problems form a natural abstraction to a range of quantitative reasoning skills that are often required of an autonomous agent to behave intelligently, such as understanding
financial news, calculating change in everyday transactions, and analyzing productivity in an organization. Solving such problems remains a key challenge for artificial intelligence (AI)
today, as they involve recognizing how goals, entities and quantities in the real-world map into a mathematical formalization - [source](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6866023.pdf)).

4. [Solving math word problems with process-and outcome-based feedback](https://arxiv.org/pdf/2211.14275)
   - **Comment:** CoT(chain of thought) and its variants such as CoT-SC, ToT elicit reasoning ability only in really large language models of size 175B, 540B even 1.8TB.
   - This is the first paper (afaik), demonstrating that using RL improves reasoning performance even for comparatively smaller language models
   - Followed by, OpenAI also released a similar [paper](https://arxiv.org/pdf/2305.20050) on the MATH dataset
   - Interestingly, GPT-4 O1 (likely) used this RL approach albeit with smaller models

5. [REDUCING ACTIVATION RECOMPUTATION IN LARGE TRANSFORMER MODELS](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf)
   - Activation memory scales up quadratically and encompasses a large amount of VRAM
   - This paper provides a detailed breakdown of how to optimize the activation memory requirement under different training settings (parallelism)

## Blogs
1. [Deep Learning Is Hitting a Wall](https://nautil.us/deep-learning-is-hitting-a-wall-238440/) - 2022
   - [Tweet](https://x.com/karpathy/status/1971220449515516391) by Andrej Karpathy related to the article - 2025
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
   - Though you freeze the parameters for some layers, it is a good practice to pass a filter to the optimizer , ```torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters())```. Otherwise, the optimiser will allocate memory to all parameters, although the gradients won't be computed for those parameters. This will save significant memory especially for really large models.
 2. What if the model underperforms?
     - Keep the batch size  small during the initial iterations and increase the size as the iterations increase (adjust warm-up lr accordingly)
     - Use hooks to debug the gradients, activation distribution. If you are using DDP, try dynamic tanh activation by removing batch/layer normalisation and see if the model performance improves
     - In LLMs, you may also vary the beam size while dealing with reasoning models. For example, page 6 in [the paper](https://arxiv.org/pdf/2103.03874)
       
 3. ERROR TYPE: SEMANTIC
     - Reason: Be mindful while using Compose functions (such as `nn.sequential, compose,..`)
     - We easily overlook the order of execution when we use compositional functions, such as the `compose` function in torchvision
     - It introduces subtle performance degradation (which is difficult to debug)
     - Say, you want to carry out **ablation studies**
     - Assume that your model requires an image in 224x224. However, the image you have is 32x32.
        - [resize --> ablation operations] is completely different from
        - [ablation operations --> resize]
        - Though both works (semantic error), they are totally two different things!
        - The pre-processing pipeline is completely different
        - Order does matter!
 5. ERROR TYPE: SEMANTIC,
     - Be aware that " Gradescaler" from PyTorch is required only when you use "float16" not when using "bfloat16" as "bfloat16" has enough dynamic range!
 6. Use Pytorch's in-built `vmap` for vectorising the operation over batch [doc](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html)
 7. Error Type: Semantic
     - Using ```nn.Sequential(nn.Linear)``` instead of just `nn.Linear` makes no difference operation-wise. However, we often forget to replicate the same architecture in THE SAME WAY during inference.
     - For example, using `nn.Sequential(nn.Linear)` while training and `nn.Linear` while evaluating won't raise an error. We may find this out if the performance drops significantly!
## Careful Setup Instructions
1. After setting up the conda environment and activating it
    - Check the `which python` and ensure it is pointing to the current environment
    - Use `python -m pip install torch torchvision torchaudio` that safely uses the `pip` package from the current environment not from other existing environments by mistake.
3. If you are on a shared cluster (servers), you should be careful and ensure every step you follow is correct. Do not ASSUME anything
    - For example, I set up a conda environment and installed all required packages.
    - I then used `ipython` for initial experimentation (that is the only option I have due to security concerns).
    - Everything was running smoothly, so I didn't doubt anything.

    - One day, the code threw an error. I spent half a day going through the documentation to fix it.
    - Nothing worked!
    - Finally, I realiszd, I didn't install `ipython` at all in `my environment`, therefore, it used the `ipython` from the base environment (sic!)
    - The base environment has a different PyTorch version than I intended!
4. Sometimes the issue is not related to your code. However, you encounter errors related to the CUDA driver (especially on shared clusters).
    - Attempt clearing the cache directory (`.cache/torch/kernels`) as discussed [here](https://discuss.pytorch.org/t/torch-prod-produces-runtimeerror-cuda-driver-error-invalid-argument/179054/29)
