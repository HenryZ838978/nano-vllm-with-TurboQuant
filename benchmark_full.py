"""
Comprehensive benchmark: nano-vllm + TurboQuant on MiniCPM4.1-8B
Each test runs in a fresh subprocess to avoid NCCL process group issues.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import sys
import json
import subprocess

MODEL_PATH = "/cache/zhangjing/voiceagent/models/MiniCPM4.1-8B"
PYTHON = "/cache/zhangjing/miniconda3/envs/voiceagent/bin/python"

QUALITY_PROMPTS = [
    "What is KV cache quantization and why does it matter for LLM inference?",
    "Write a Python function to merge two sorted lists.",
    "Explain TCP vs UDP in 3 sentences.",
    "What are the main differences between Python 2 and Python 3?",
]


def run_single_test(test_type, kv_quant_bits=None, **kwargs):
    """Spawn a fresh process for one test, return JSON result."""
    script = f'''
import os, sys, time, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
sys.path.insert(0, "{os.getcwd()}")
from random import seed as rseed, randint
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "{MODEL_PATH}"
test_type = "{test_type}"
kv_quant_bits = {kv_quant_bits}
kwargs = {json.dumps(kwargs)}

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

if test_type == "throughput":
    num_seqs = kwargs["num_seqs"]
    input_len = kwargs["input_len"]
    output_len = kwargs["output_len"]
    max_ctx = input_len + output_len + 64
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=max_ctx, kv_quant_bits=kv_quant_bits)
    rseed(42)
    prompt_ids = [[randint(100, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)] * num_seqs
    llm.generate([[1,2,3]], SamplingParams(temperature=0.6, max_tokens=4), use_tqdm=False)
    t0 = time.time()
    llm.generate(prompt_ids, sp, use_tqdm=False)
    elapsed = time.time() - t0
    total_out = num_seqs * output_len
    throughput = total_out / elapsed
    peak = torch.cuda.max_memory_allocated() / 1024**3
    tag = f"TQ-{{kv_quant_bits}}b" if kv_quant_bits else "FP16"
    result = {{"tag": tag, "num_seqs": num_seqs, "input_len": input_len,
               "output_len": output_len, "throughput": round(throughput, 1),
               "elapsed": round(elapsed, 2), "peak_gb": round(peak, 2)}}
    llm.exit()

elif test_type == "quality":
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=2048, kv_quant_bits=kv_quant_bits)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts_raw = {json.dumps(QUALITY_PROMPTS)}
    prompts = [tokenizer.apply_chat_template(
        [{{"role": "user", "content": p}}], tokenize=False, add_generation_prompt=True
    ) for p in prompts_raw]
    sp = SamplingParams(temperature=0.01, max_tokens=200)
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    tag = f"TQ-{{kv_quant_bits}}b" if kv_quant_bits else "FP16"
    result = {{"tag": tag,
               "outputs": [o["text"][:500] for o in outputs],
               "token_counts": [len(o["token_ids"]) for o in outputs]}}
    llm.exit()

print("__RESULT__" + json.dumps(result, ensure_ascii=False))
'''
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4"
    proc = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
        cwd=os.getcwd(),
    )
    for line in proc.stdout.split("\n"):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    print(f"  STDERR: {proc.stderr[-500:]}", file=sys.stderr)
    raise RuntimeError(f"Test failed: {test_type} bits={kv_quant_bits}")


def memory_analysis():
    num_layers = 32
    num_kv_heads = 2
    head_dim = 128
    dtype_bytes = 2

    per_token_fp16 = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    per_token_k_fp16 = num_layers * num_kv_heads * head_dim * dtype_bytes
    per_token_v_fp16 = per_token_k_fp16

    mse_bits, qjl_bits = 2, 1
    per_token_k_tq3 = num_layers * num_kv_heads * (
        head_dim * mse_bits // 8 + head_dim * qjl_bits // 8 + 4
    )
    per_token_tq3 = per_token_k_tq3 + per_token_v_fp16

    avail = 5.0 * 1024**3
    return {
        "per_token_fp16": per_token_fp16,
        "per_token_tq3": per_token_tq3,
        "ratio_overall": round(per_token_fp16 / per_token_tq3, 2),
        "ratio_k_only": round(per_token_k_fp16 / per_token_k_tq3, 2),
        "max_tok_fp16_5gb": int(avail / per_token_fp16),
        "max_tok_tq3_5gb": int(avail / per_token_tq3),
    }


def main():
    print("=" * 70)
    print("  nano-vllm + TurboQuant — Full Benchmark")
    print("  Model: MiniCPM4.1-8B (8B, GQA 32Q/2KV, head_dim=128)")
    print("  GPU: NVIDIA RTX 4090 24GB")
    print("=" * 70)

    all_results = {}

    # --- Throughput ---
    configs = [
        {"label": "bs1_short", "num_seqs": 1, "input_len": 128, "output_len": 128},
        {"label": "bs1_long",  "num_seqs": 1, "input_len": 512, "output_len": 256},
        {"label": "bs8_short", "num_seqs": 8, "input_len": 128, "output_len": 128},
        {"label": "bs8_long",  "num_seqs": 8, "input_len": 512, "output_len": 128},
    ]

    print("\n### Throughput Tests")
    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n  [{label}] batch={cfg['num_seqs']}, in={cfg['input_len']}, out={cfg['output_len']}")
        for bits in [None, 3]:
            tag_key = f"TQ-{bits}b" if bits else "FP16"
            print(f"    Running {tag_key}...", end=" ", flush=True)
            r = run_single_test("throughput", kv_quant_bits=bits, **cfg)
            all_results[f"{label}_{tag_key}"] = r
            print(f"{r['throughput']} tok/s, {r['peak_gb']} GB")
        cfg["label"] = label

    # --- Quality ---
    print("\n### Quality Tests")
    for bits in [None, 3]:
        tag_key = f"TQ-{bits}b" if bits else "FP16"
        print(f"  Running {tag_key}...", end=" ", flush=True)
        r = run_single_test("quality", kv_quant_bits=bits)
        all_results[f"qual_{tag_key}"] = r
        print(f"done ({sum(r['token_counts'])} tokens)")

    # --- Memory ---
    mem = memory_analysis()
    all_results["memory"] = mem

    # --- Print README table ---
    print("\n" + "=" * 70)
    print("  RESULTS (copy-paste for README)")
    print("=" * 70)

    print("\n## Decode Throughput\n")
    print("| Config | Batch | Input | Output | FP16 tok/s | TQ-3b tok/s | Ratio |")
    print("|--------|-------|-------|--------|------------|-------------|-------|")
    for cfg in configs:
        label = cfg["label"]
        fp = all_results.get(f"{label}_FP16", {})
        tq = all_results.get(f"{label}_TQ-3b", {})
        if fp and tq:
            ratio = f"{tq['throughput']/fp['throughput']:.2f}x"
            print(f"| {label} | {fp['num_seqs']} | {fp['input_len']} | {fp['output_len']} | {fp['throughput']} | {tq['throughput']} | {ratio} |")

    print(f"\n## Memory Analysis (Theoretical — Bit-Packed TQ-3bit)\n")
    print(f"| Metric | FP16 | TQ-3bit | Improvement |")
    print(f"|--------|------|---------|-------------|")
    print(f"| KV bytes/token | {mem['per_token_fp16']} | {mem['per_token_tq3']} | {mem['ratio_overall']}x |")
    print(f"| K-only bytes/token | {mem['per_token_fp16']//2} | {mem['per_token_tq3'] - mem['per_token_fp16']//2} | {mem['ratio_k_only']}x |")
    print(f"| Max tokens (5GB budget) | {mem['max_tok_fp16_5gb']:,} | {mem['max_tok_tq3_5gb']:,} | {mem['max_tok_tq3_5gb']/mem['max_tok_fp16_5gb']:.1f}x |")

    print(f"\n## Output Quality (temperature≈0, same prompts)\n")
    fp_q = all_results.get("qual_FP16", {})
    tq_q = all_results.get("qual_TQ-3b", {})
    for i, prompt in enumerate(QUALITY_PROMPTS):
        print(f"**Q: {prompt}**\n")
        for tag, qr in [("FP16", fp_q), ("TQ-3b", tq_q)]:
            out = qr["outputs"][i] if qr else "N/A"
            if "</think>" in out:
                out = out.split("</think>", 1)[1].strip()
            print(f"*{tag}*: {out[:250]}...\n")

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nFull results → benchmark_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
