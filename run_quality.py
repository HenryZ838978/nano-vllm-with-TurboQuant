"""Quality comparison: FP16 vs TQ-3b side-by-side."""
import os, json, subprocess, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

MODEL = "/cache/zhangjing/voiceagent/models/MiniCPM4.1-8B"
PYTHON = "/cache/zhangjing/miniconda3/envs/voiceagent/bin/python"

PROMPTS = [
    "What is KV cache quantization and why does it matter for LLM inference?",
    "Write a Python function to merge two sorted lists.",
    "Explain TCP vs UDP in 3 sentences.",
    "What are the main differences between Python 2 and Python 3?",
]

def run_quality(bits, port):
    prompts_json = json.dumps(PROMPTS)
    script = f'''
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "4"; os.environ["NCCL_PORT"] = "{port}"
import json, sys; sys.path.insert(0, "{os.getcwd()}")
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "{MODEL}"
llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1, max_model_len=2048, kv_quant_bits={bits})
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompts_raw = {prompts_json}
prompts = [tokenizer.apply_chat_template(
    [{{"role": "user", "content": p}}], tokenize=False, add_generation_prompt=True
) for p in prompts_raw]
sp = SamplingParams(temperature=0.01, max_tokens=200)
outputs = llm.generate(prompts, sp, use_tqdm=False)
results = []
for o in outputs:
    text = o["text"]
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    results.append(text[:300])
print("__RESULT__" + json.dumps(results, ensure_ascii=False))
llm.exit()
'''
    proc = subprocess.run([PYTHON, "-c", script], capture_output=True, text=True, timeout=300, cwd=os.getcwd())
    for line in proc.stdout.split("\n"):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    print(f"STDERR: {proc.stderr[-300:]}")
    return None

print("Running FP16...")
fp16 = run_quality(None, 31000)
print("Running TQ-3b...")
tq3 = run_quality(3, 31001)

print("\n" + "=" * 70)
print("QUALITY COMPARISON")
print("=" * 70)
for i, prompt in enumerate(PROMPTS):
    print(f"\nQ: {prompt}")
    print(f"\n  [FP16]:  {fp16[i][:250] if fp16 else 'FAIL'}")
    print(f"\n  [TQ-3b]: {tq3[i][:250] if tq3 else 'FAIL'}")
    print()

with open("quality_results.json", "w") as f:
    json.dump({"fp16": fp16, "tq3": tq3, "prompts": PROMPTS}, f, indent=2, ensure_ascii=False)
print("Saved to quality_results.json")
