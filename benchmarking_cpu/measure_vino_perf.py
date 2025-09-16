import time
import torch
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

from datetime import datetime
import os

LOG_DIR = "bench_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "openvino_gemma2b.txt")

def save_prompt_and_completion_ov(tokenizer, prompt_ids, out_sequences, path=LOG_PATH, run_idx=0):
    # out_sequences: model.generate(...).sequences or a plain tensor
    if hasattr(out_sequences, "sequences"):
        out_ids = out_sequences.sequences
    else:
        out_ids = out_sequences
    new_ids = out_ids[0, prompt_ids.shape[-1]:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}Z] run={run_idx}\n")
        f.write("PROMPT:\n")
        f.write(prompt_text + "\n")
        f.write("COMPLETION:\n")
        f.write(completion + "\n")
        f.write("-" * 40 + "\n")

        

MODEL_ID = "google/gemma-2b"  # or "google/gemma-2-2b"
DEVICE = "CPU"  # or "GPU.0" for Intel iGPU/ARC
OV_CONFIG = {
    "PERFORMANCE_HINT": "LATENCY",  # or "THROUGHPUT"
    # "NUM_STREAMS": "1",  # for CPU; tune as needed
}
PROMPT = "Python is a programming language that is"
MAX_NEW_TOKENS = 128
NUM_RUNS = 5
WARMUP = 1

def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = OVModelForCausalLM.from_pretrained(
        MODEL_ID,
        export=True,        # export to OpenVINO IR on the fly
        device=DEVICE,
        ov_config=OV_CONFIG
    )
    return tok, model

def generate_stream_first_token(model, input_ids, attention_mask, max_new_tokens, tok):
    # OpenVINO model.generate returns output ids; it does not natively stream characters.
    # To approximate TTFT, we measure until first new token is materialized by greedy generation step loop.
    # For better fidelity, one can call step-by-step with past_key_values. This loop simulates that.
    start = time.perf_counter()
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=False
    )
    end = time.perf_counter()
    # We cannot directly observe the very first token time without token-level stepping.
    # As an approximation, we estimate TTFT by measuring a short generation with max_new_tokens=1.
    return output_ids.sequences, start, end

def measure(tok, model):
    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    # Warmup
    for _ in range(WARMUP):
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4)

    ttft_list, gen_time_list, tokps_list, out_tokens_list = [], [], [], []

    # Approximate TTFT by a separate 1-token run
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1, do_sample=False, use_cache=True)
        t1 = time.perf_counter()
        ttft = t1 - t0

        g0 = time.perf_counter()
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        save_prompt_and_completion_ov(tok, input_ids, out, run_idx=_)
        g1 = time.perf_counter()
        total_time = g1 - g0

        new_tokens = out.shape[-1] - input_ids.shape[-1]
        tokps = new_tokens / total_time if total_time > 0 else float("nan")

        ttft_list.append(ttft)
        gen_time_list.append(total_time)
        tokps_list.append(tokps)
        out_tokens_list.append(new_tokens)

    print("OpenVINO Benchmark")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"OV config: {OV_CONFIG}")
    print(f"Input tokens: {input_ids.shape[-1]}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Runs: {NUM_RUNS}")
    print(f"TTFT approx (s): {[round(x, 4) for x in ttft_list]}")
    print(f"Latency total (s): {[round(x,4) for x in gen_time_list]}")
    print(f"Throughput (tok/s): {[round(x,2) for x in tokps_list]}")
    print(f"Generated tokens: {out_tokens_list}")

if __name__ == "__main__":
    tok, model = load()
    measure(tok, model)
