# DSDE: Dynamic Speculative Decoding Engine

**Based on vLLM v0.9.4**

This repository integrates our research and development on **speculative decoding for LLM serving**

1. **Dynamic Speculative Length (DSL) Adaptation**

   - Sequence- and iteration-level draft length control (DSDE core).
   - Training-free framework guided by stability signals (KLD variance, acceptance rate analysis).
   - Related publication and upstream discussion:
     - [DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving (arXiv:2509.01083)](https://arxiv.org/abs/2509.01083)
     - vLLM GitHub Issue: [#17984 – Dynamic speculative length adaptation](https://github.com/vllm-project/vllm/issues/17984)
     - Per-sequence decoding: [Spec-Decode] Add DynamicProposer for per-sequence dynamic speculative decoding #26504 (https://github.com/vllm-project/vllm/pull/26504)

2. **Tree-Attention Speculative Decoding (RA Contribution)**
   - This enables efficient speculative decoding by avoiding batch expansion, providing memory-efficient attention computation for tree-structured token generation patterns commonly used in speculative sampling.
   - Prior works such as _SpecInfer_ have demonstrated the effectiveness of tree-attention, but kernel optimization has remained limited. Our contribution provides a practical and optimized implementation.
   - Related work and upstream discussion:
     - Tree-attention Issue: [#18327 – Add support for tree-attention masks](https://github.com/vllm-project/vllm/issues/18327)
     - Tree-attention in FlashAttention PR: [#81 – Tree-attention mask support](https://github.com/vllm-project/flash-attention/pull/81)

---

🔗 **Note**: This is an internal repository. External open-source contributions are referenced separately via the linked issues and PRs.
