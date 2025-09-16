import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from datetime import datetime
import os
# For logging output tokens
LOG_DIR = "bench_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "cpu_pytorch_gemma2b.txt")

def save_prompt_and_completion(tokenizer, prompt_ids, output_ids, path=LOG_PATH, run_idx=0):
    # Decode only new tokens (completion)
    new_ids = output_ids[0, prompt_ids.shape[-1]:]    
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)    
    prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
    
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}Z] run={run_idx}\n")
        f.write("PROMPT:\n")
        f.write(prompt_text + "\n")
        f.write("COMPLETION:\n")
        f.write(completion + "\n")
        f.write("-" * 40 + "\n")
        
#specs
MODEL_ID = "google/gemma-2b"  # or "google/gemma-2-2b"
DEVICE = "cpu"
DTYPE = torch.float32
PROMPT = "Explain the difference between throughput and latency in LLM inference."
MAX_NEW_TOKENS = 128
NUM_RUNS = 5
WARMUP = 1

def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=None
    )
    model = model.to(DEVICE)
    model.eval()
    return tok, model

def measure(tok, model):
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    # Warmup
    for _ in range(WARMUP):
        _ = model.generate(input_ids, max_new_tokens=4)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ttft_list, gen_time_list, tokps_list, out_tokens_list = [], [], [], []
    for _ in range(NUM_RUNS):
        # Streamer to capture first token timing
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        start_prefill = time.perf_counter()
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            streamer=streamer,
            use_cache=True,
        )

        # Start generation in a background thread to capture first token arrival
        import threading
        first_token_time = {"t": None}
        def read_stream():
            for i, _ch in enumerate(streamer):
                if i == 0 and first_token_time["t"] is None:
                    first_token_time["t"] = time.perf_counter()

        t = threading.Thread(target=read_stream)
        t.start()
        out_ids = model.generate(**gen_kwargs)
        
        save_prompt_and_completion(tok, input_ids, out_ids, run_idx=_)
        t.join()
        end_all = time.perf_counter()

        # Compute metrics
        ttft = (first_token_time["t"] - start_prefill) if first_token_time["t"] else None
        total_time = end_all - start_prefill
        # Count new tokens
        new_tokens = (out_ids.shape[-1] - input_ids.shape[-1])
        gen_time = total_time  # includes prefill; report overall latency
        tokps = new_tokens / gen_time if gen_time > 0 else float("nan")

        ttft_list.append(ttft)
        gen_time_list.append(gen_time)
        tokps_list.append(tokps)
        out_tokens_list.append(new_tokens)

    print("CPU PyTorch Benchmark")
    print(f"Model: {MODEL_ID}")
    print(f"Input tokens: {input_ids.shape[-1]}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Runs: {NUM_RUNS}")
    print(f"TTFT (s): {[round(x, 4) if x is not None else None for x in ttft_list]}")
    print(f"Latency total (s): {[round(x,4) for x in gen_time_list]}")
    print(f"Throughput (tok/s): {[round(x,2) for x in tokps_list]}")
    print(f"Generated tokens: {out_tokens_list}")

if __name__ == "__main__":
    tok, model = load()
    measure(tok, model)
