"""Test nano-vllm with MiniCPM4.1-8B: FP16 baseline then TurboQuant-3bit."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_PATH = "/cache/zhangjing/voiceagent/models/MiniCPM4.1-8B"


def run_test(kv_quant_bits=None):
    tag = f"TQ-{kv_quant_bits}bit" if kv_quant_bits else "FP16"
    print(f"\n{'='*60}")
    print(f"  Testing: {tag}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        kv_quant_bits=kv_quant_bits,
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
    prompts = [
        "What is KV cache quantization in large language models?",
        "Write a short Python function that computes fibonacci numbers.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt[:80]}...")
        print(f"Output: {output['text'][:300]}")

    llm.exit()
    del llm
    import torch
    torch.cuda.empty_cache()
    print(f"\n[{tag}] Done.\n")


if __name__ == "__main__":
    print("Step 1: FP16 baseline")
    run_test(kv_quant_bits=None)

    print("Step 2: TurboQuant 3-bit")
    run_test(kv_quant_bits=3)
