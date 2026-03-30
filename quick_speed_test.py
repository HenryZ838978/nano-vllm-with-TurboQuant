"""Quick speed comparison: FP16 vs TQ-3b optimized."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time, torch, json, subprocess, sys
from random import randint as _ri

MODEL = "/cache/zhangjing/voiceagent/models/MiniCPM4.1-8B"
PYTHON = sys.executable
_port_counter = [30000]

def run_test(kv_quant_bits, batch, input_len, output_len):
    _port_counter[0] += 1
    port = _port_counter[0]
    script = f'''
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "4"; os.environ["NCCL_PORT"] = "{port}"
import time, torch, json
from random import seed as rseed, randint
import sys; sys.path.insert(0, "{os.getcwd()}")
from nanovllm import LLM, SamplingParams

llm = LLM("{MODEL}", enforce_eager=True, tensor_parallel_size=1,
          max_model_len={input_len + output_len + 64}, kv_quant_bits={kv_quant_bits})
rseed(42)
prompts = [[randint(100,10000) for _ in range({input_len})] for _ in range({batch})]
sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens={output_len})] * {batch}
llm.generate([[1,2,3]], SamplingParams(temperature=0.6, max_tokens=4), use_tqdm=False)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
llm.generate(prompts, sp, use_tqdm=False)
e = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**3
tps = {batch} * {output_len} / e
print(f"RESULT|{{tps:.1f}}|{{peak:.2f}}|{{e:.2f}}")
llm.exit()
'''
    proc = subprocess.run([PYTHON, "-c", script], capture_output=True, text=True, timeout=300)
    for line in proc.stdout.split("\n"):
        if line.startswith("RESULT|"):
            parts = line.split("|")
            return float(parts[1]), float(parts[2]), float(parts[3])
    print("STDERR:", proc.stderr[-300:])
    return None, None, None


configs = [
    (1,  128, 128),
    (8,  128, 128),
    (1,  512, 256),
    (8,  512, 128),
]

print(f"{'Config':<30} {'FP16 tok/s':>12} {'TQ-3b tok/s':>12} {'Ratio':>8} {'FP16 GB':>10} {'TQ-3b GB':>10}")
print("-" * 85)

for batch, inp, out in configs:
    label = f"bs={batch}, in={inp}, out={out}"
    fp_tps, fp_mem, _ = run_test(None, batch, inp, out)
    tq_tps, tq_mem, _ = run_test(3, batch, inp, out)
    if fp_tps and tq_tps:
        ratio = f"{tq_tps/fp_tps:.2f}x"
        print(f"{label:<30} {fp_tps:>12.1f} {tq_tps:>12.1f} {ratio:>8} {fp_mem:>10.2f} {tq_mem:>10.2f}")
    else:
        print(f"{label:<30} {'FAIL':>12} {'FAIL':>12}")
