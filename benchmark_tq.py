"""
Benchmark: FP16 vs TurboQuant KV cache on MiniCPM4.1-8B via nano-vllm.
Compares output quality (side-by-side), decode throughput, and GPU memory.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import time
import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "/cache/zhangjing/voiceagent/models/MiniCPM4.1-8B"

PROMPTS = [
    "What is KV cache quantization and why does it matter for LLM inference?",
    "Write a Python function to merge two sorted lists into one sorted list.",
    "Explain the difference between TCP and UDP in simple terms.",
]


def run_benchmark(kv_quant_bits=None):
    tag = f"TQ-{kv_quant_bits}bit" if kv_quant_bits else "FP16"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        kv_quant_bits=kv_quant_bits,
    )

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    sp = SamplingParams(temperature=0.7, max_tokens=200)

    t0 = time.time()
    outputs = llm.generate(chat_prompts, sp, use_tqdm=False)
    elapsed = time.time() - t0

    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    llm.exit()
    del llm
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "elapsed": elapsed,
        "total_tokens": total_tokens,
        "throughput": total_tokens / elapsed,
        "peak_mem_gb": peak_mem,
        "outputs": outputs,
    }


def main():
    print("=" * 70)
    print("  nano-vllm + TurboQuant Benchmark on MiniCPM4.1-8B")
    print("=" * 70)

    results = {}

    print("\n[1/2] Running FP16 baseline...")
    results["fp16"] = run_benchmark(kv_quant_bits=None)

    print("[2/2] Running TurboQuant 3-bit...")
    results["tq3"] = run_benchmark(kv_quant_bits=3)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for key in ["fp16", "tq3"]:
        r = results[key]
        print(f"\n  [{r['tag']}]")
        print(f"    Total tokens: {r['total_tokens']}")
        print(f"    Time:         {r['elapsed']:.2f}s")
        print(f"    Throughput:   {r['throughput']:.1f} tok/s")
        print(f"    Peak GPU mem: {r['peak_mem_gb']:.2f} GB")

    print("\n" + "-" * 70)
    print("  OUTPUT COMPARISON (side-by-side)")
    print("-" * 70)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {i+1}: {prompt[:60]}...")
        for key in ["fp16", "tq3"]:
            r = results[key]
            text = r["outputs"][i]["text"]
            lines = text.strip().split("\n")
            preview = "\n    ".join(lines[:6])
            if len(lines) > 6:
                preview += "\n    ..."
            print(f"\n    [{r['tag']}]:")
            print(f"    {preview}")

    print("\n" + "=" * 70)
    fp = results["fp16"]
    tq = results["tq3"]
    speedup = fp["throughput"] / tq["throughput"]
    print(f"  Summary: TQ-3bit decode is {speedup:.2f}x slower than FP16")
    print(f"           (expected — PyTorch-level gather+compress, no Triton kernel yet)")
    print(f"  Next step: Triton fused TQ attention kernel for parity/speedup")
    print("=" * 70)


if __name__ == "__main__":
    main()
