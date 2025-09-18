# Documenting learnings, ideas, papers, blogs and scripts for my reference

## Cool Ideas
1. [MegaKernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
    - Comment: If we can fuse a single attention module, why not fuse the entire model to get low latency?
## Mixed Precision training
1. Users **should not** manually cast their model or data to bf16
2. [Best practices guide](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

## Papers
1. [Emergent Abilities of Large Language Models](https://arxiv.org/pdf/2206.07682)
    - Comment: What really happens when we scale the model? In-context learning ( great sample efficiency) is a commonly known ability.
    - Going through the paper gives us deeper insights (say, we can sense that **test-time scaling** is worth trying (back in 2022) without even using our compute)
    - The quote "Emergence is when _quantitative_ changes in a system result in _qualitative_ changes in behaviour" cited in the paper deeply resonates with me
