# Tensors, Einsum &amp; Modern Tricks

Deck 12 of the [Linear Algebra for AI / ML](https://github.com/BrendanJamesLynskey/LLM_Hub_Linear_Algebra) series.

**Live presentation:** https://brendanjameslynskey.github.io/Linear_Algebra_AI_12_Tensors_Einsum_and_Modern_Tricks/

When you have batch, head, sequence, head-dim and feature, matrix algebra runs out of letters. Tensors, einsum and named indices give you the language to write &mdash; and parallelise &mdash; the algorithms that move large models across thousands of GPUs.

## What's inside

- From matrix to tensor: 3D and 4D arrays in real ML code
- Einsum: contraction, broadcasting, permutation in one notation
- Reading a real attention block in einsum (six lines, no transpose calls)
- Strides, views, contiguous-vs-not, BHLD vs BLHD layouts
- FlashAttention as block matmul with online softmax
- Mixture of Experts (MoE) as a sparse-projection FFN; Mixtral 8&times;7B numbers
- Data / Tensor / Pipeline parallelism &mdash; the three orthogonal sharding axes
- GSPMD: sharding an einsum is a property of the index pattern
- Where the series lands and links to companion sub-hubs

The closing deck of the series &mdash; ties the linear-algebra foundations back to the systems hubs (NVIDIA GPUs, Google TPUs, CUDA, Modern Architectures).

Single-page HTML, KaTeX-rendered maths, no build step. Open `index.html` directly.
