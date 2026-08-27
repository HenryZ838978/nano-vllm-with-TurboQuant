> ✅ **Status: Done — archived on 2026-08-27.** This integration is feature-complete and no longer maintained.

# nano-vLLM + TurboQuant

Integrating TurboQuant KV cache compression into [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), a lightweight LLM inference engine.

**TurboQuant** (ICLR 2026) is a two-stage vector quantization algorithm that compresses KV cache keys with near-optimal distortion, enabling longer context and higher concurrency on memory-constrained GPUs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  nano-vLLM Engine                                               │
│  ┌───────────┐   ┌─────────────┐   ┌────────────────────────┐  │
│  │ Scheduler  │──▶│ ModelRunner │──▶│ MiniCPM4.1-8B Model    │  │
│  │ + Paging   │   │ KV Alloc    │   │ (LongRoPE, GQA, μP)   │  │
│  └───────────┘   └─────┬───────┘   └──────────┬─────────────┘  │
│                         │                      │                │
│                  ┌──────▼──────┐        ┌──────▼──────┐         │
│                  │ FP16 Paged  │        │ TQ Compressed│         │
│                  │ KV Cache    │        │ K Cache      │         │
│                  │ (K + V)     │        │ (k_mse,signs,│         │
│                  │             │        │  residual_norm)        │
│                  └──────┬──────┘        └──────┬──────┘         │
│                         │                      │                │
│                  ┌──────▼──────────────────────▼──────┐         │
│                  │         Attention Layer             │         │
│                  │  Prefill: FlashAttention            │         │
│                  │  Decode:  TQ Asymmetric Estimator   │         │
│                  │          or FlashAttention (FP16)   │         │
│                  └────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### TurboQuant Two-Stage Compression

For each key vector **k** ∈ ℝᵈ:

1. **Stage 1 — MSE-Optimal Quantization**: Random rotation Π, then per-coordinate Lloyd-Max quantization on the normalized vector. Produces **k\_mse** — the MSE-optimal reconstruction.

2. **Stage 2 — QJL Sign Correction**: The residual **r = k − k\_mse** is projected through a random matrix **S**, and only the 1-bit signs are kept. This provides an unbiased correction to the inner product estimate.

**Asymmetric Attention Estimator** (no full decompression needed):

```
⟨q, k⟩ ≈ ⟨q, k_mse⟩ + ‖r_k‖ · √(π/2) / m · ⟨Sq, sign(Sr_k)⟩
```

This computes attention scores directly from compressed keys, combining the MSE reconstruction term with the QJL correction term.

## Benchmark Results

**Hardware**: NVIDIA RTX 4090 24GB  
**Model**: MiniCPM4.1-8B (8B params, GQA 32Q/2KV, head\_dim=128, 32 layers)  
**Software**: nano-vLLM, enforce\_eager=True, single GPU

### Decode Throughput

| Config | Batch | Input Len | Output Len | FP16 (tok/s) | TQ-3bit (tok/s) | Ratio |
|--------|-------|-----------|------------|-------------|-----------------|-------|
| Short, single | 1 | 128 | 128 | 26.3 | 10.5 | 0.40x |
| Long, single | 1 | 512 | 256 | 26.1 | 10.4 | 0.40x |
| Short, batch | 8 | 128 | 128 | 187.4 | 78.1 | 0.42x |
| Long, batch | 8 | 512 | 128 | 178.3 | 77.3 | 0.43x |

**Analysis**: TQ-3bit currently runs at ~0.40–0.43x FP16 decode speed. The gap is due to the decode path implementation:
- **FP16**: FlashAttention — a highly optimized fused CUDA kernel that reads paged KV cache directly
- **TQ-3bit**: PyTorch-level gather + two matmuls (MSE term + QJL correction) + softmax + output matmul

This is an algorithmic limitation of the current implementation, not the TurboQuant algorithm itself. See [Roadmap](#roadmap) for the Triton kernel path.

### Peak GPU Memory

| Config | FP16 (GB) | TQ-3bit (GB) | Delta |
|--------|----------|-------------|-------|
| bs=1, short | 19.35 | 22.38 | +3.03 |
| bs=8, long | 19.69 | 22.73 | +3.04 |

The current implementation stores **both** the FP16 KV cache (required for prefill FlashAttention) and the compressed K cache (for TQ decode). The +3 GB overhead is the compressed K caches across 32 layers.

### Theoretical Memory Savings (Bit-Packed TQ-3bit)

With bit-packed storage (replacing FP16 K cache entirely):

| Metric | FP16 | TQ-3bit (bit-packed) | Improvement |
|--------|------|---------------------|-------------|
| K bytes/token | 16,384 | 3,328 | **4.92x** |
| Total KV bytes/token | 32,768 | 19,712 | **1.66x** |
| Max tokens (5 GB budget) | 163,840 | 272,357 | **1.66x** |

**What this means in practice** (MiniCPM4.1-8B on RTX 4090):

| Scenario | FP16 | TQ-3bit (projected) |
|----------|------|---------------------|
| Max context length (single seq) | ~32K tokens | ~53K tokens |
| Concurrent 4K-context sessions | ~8 sessions | ~13 sessions |
| 8K-context sessions | ~4 sessions | ~6 sessions |

> The value isn't "faster per-token" — it's "the GPU that could only load the model can now also **serve** it."

### Output Quality

Near-greedy sampling (temperature=0.01), identical prompts. Both modes produce coherent, factually correct responses.

<details>
<summary><b>Q: What is KV cache quantization and why does it matter?</b></summary>

**FP16**: *"I need to figure out what KV cache quantization is... KV stands for Key-Value... KV cache refers to the cache used during the inference process. When an LLM processes..."*

**TQ-3bit**: *"We are going to discuss KV cache quantization and its importance for LLM inference... the KV cache refers to the key-value cache used during the attention mechanism. Specifically, in the Transformer..."*

Both correctly identify KV cache, its role in attention, and the memory bottleneck.
</details>

<details>
<summary><b>Q: Write a Python function to merge two sorted lists</b></summary>

**FP16**: *"We can use two pointers, one for each list, and compare the elements at these pointers. We'll traverse both lists and at each step, pick the smaller element..."*

**TQ-3bit**: *"Since the lists are sorted, we can use a two-pointer technique. We'll traverse both lists simultaneously, comparing the current elements..."*

Both produce correct two-pointer merge implementations.
</details>

<details>
<summary><b>Q: Explain TCP vs UDP in 3 sentences</b></summary>

**FP16**: *"TCP: A reliable, connection-oriented protocol that establishes a dedicated end-to-end connection, guarantees data delivery, ensures order, and performs error checking."*

**TQ-3bit**: *"TCP establishes a reliable, ordered connection between apps, ensuring all data is delivered accurately and in sequence."*

Both accurately distinguish TCP (reliable, ordered) from UDP (lightweight, connectionless).
</details>

<details>
<summary><b>Q: Main differences between Python 2 and Python 3?</b></summary>

**FP16**: *"Print Statement vs Print Function: Python 2: print is a statement... Python 3: print() is a function..."*

**TQ-3bit**: *"In Python 2, print is a statement, whereas in Python 3, print is a function..."*

Both correctly enumerate the key differences (print, division, unicode, etc.).
</details>

## Implementation Details

### Key Optimizations (v2)

1. **Persistent Compressed K Cache**: Keys are compressed once (during prefill and each decode step) and stored in dedicated paged caches. Previous version re-compressed the entire KV history every decode step — O(seq\_len) → O(1).

2. **Vectorized Gather**: Paged cache gather uses torch advanced indexing instead of Python loops — single GPU kernel per gather.

3. **GQA-Aware Attention**: The asymmetric estimator reshapes queries to (B, H\_kv, G, 1, D) and broadcasts over KV heads, avoiding the 16x `repeat_interleave` memory expansion for GQA (MiniCPM has 32 Q heads / 2 KV heads).

### Files

```
nanovllm/
├── turboquant/
│   ├── __init__.py
│   ├── compressor.py      # TurboQuantEngine: compress_keys, asymmetric_attention
│   └── lloyd_max.py        # Lloyd-Max optimal quantizer solver
├── models/
│   ├── qwen3.py            # Original Qwen3 model
│   └── minicpm.py          # MiniCPM4.1-8B (LongRoPE, μP, scale_depth)
├── layers/
│   └── attention.py        # Attention with TQ decode path
├── engine/
│   ├── model_runner.py     # KV + TQ cache allocation, model auto-detect
│   ├── llm_engine.py       # Multi-EOS support, trust_remote_code
│   └── scheduler.py        # Set-based EOS checking
└── config.py               # kv_quant_bits option
```

### Usage

```python
from nanovllm import LLM, SamplingParams

# FP16 baseline
llm = LLM("openbmb/MiniCPM4.1-8B", enforce_eager=True)

# TQ-3bit KV cache compression
llm = LLM("openbmb/MiniCPM4.1-8B", enforce_eager=True, kv_quant_bits=3)

outputs = llm.generate(["Hello!"], SamplingParams(temperature=0.7, max_tokens=256))
```

## Roadmap

- [x] TurboQuant two-stage compression (Lloyd-Max + QJL)
- [x] Asymmetric attention estimator (GQA-aware)
- [x] Persistent compressed K cache (compress-once)
- [x] MiniCPM4.1-8B model support (LongRoPE, μP scaling)
- [ ] **Triton fused TQ attention kernel** — fuse gather + decompress + Q@K + QJL correction + softmax + V matmul into a single kernel. Expected to close the 2.5x speed gap.
- [ ] **Bit-packed storage** — store 2-bit MSE indices + 1-bit QJL signs packed, replacing FP16 K cache entirely. Enables the 4.92x K compression ratio.
- [ ] **TQ-4bit mode** — 3-bit MSE + 1-bit QJL for higher fidelity at modest additional cost.
- [ ] CUDA graph support for TQ decode path

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://openreview.net/forum?id=placeholder) (ICLR 2026)
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) — Lightweight LLM inference engine
- [MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B) — Efficient 8B language model

## License

Same as the upstream nano-vLLM project.
