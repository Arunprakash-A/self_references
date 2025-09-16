import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from datetime import datetime
import os

LOG_DIR = "bench_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "gpu_pytorch_gemma2b.txt")

def save_prompt_and_completion(tokenizer, prompt_ids, output_ids, path=LOG_PATH, run_idx=0):
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
        

MODEL_ID = "google/gemma-2b"  # or "google/gemma-2-2b"
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
PROMPT = "In the field of artificial intelligence, measuring the latency of large language models is an important task for researchers, engineers, and practitioners who want to understand how efficiently a model can generate useful output under different computational conditions. Latency is often defined as the time taken between the arrival of an input and the generation of the corresponding output. For practical applications such as chatbots, document summarization, code completion, and interactive reasoning assistants, low latency is critical for ensuring a smooth and responsive user experience. High latency can frustrate users, increase abandonment rates, and make real-time deployment difficult, especially when multiple users are interacting with a system simultaneously. The purpose of this benchmark prompt is to provide a moderately long piece of input text, roughly two hundred and fifty tokens in size, so that developers can use it to test the behavior of different models across various hardware configurations. By running this same input multiple times and collecting average response times, it becomes possible to compare the efficiency of models that have been quantized to INT4, INT8, or FP16 formats, as well as uncompressed baselines. Researchers may also want to analyze whether latency remains stable across consecutive runs, or whether warm-up effects cause variations in the timing. In addition, this prompt is designed to remain semantically coherent while being long enough to stress test caching, memory bandwidth, and other bottlenecks that may appear when inference workloads are executed on CPUs, GPUs, or specialized accelerators"
MAX_NEW_TOKENS = 128
NUM_RUNS = 5
WARMUP = 1
USE_TORCH_COMPILE = False  # try True for PyTorch 2.4+ if stable for your env

def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map={"": DEVICE}
    )
    model.eval()
    if USE_TORCH_COMPILE:
        model = torch.compile(model)
    return tok, model

def measure(tok, model):
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    # Warmup
    for _ in range(WARMUP):
        _ = model.generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=True)
        torch.cuda.synchronize()

    ttft_list, gen_time_list, tokps_list, out_tokens_list = [], [], [], []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        start_prefill = time.perf_counter()
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            streamer=streamer,
            use_cache=True,
        )

        import threading
        first_token_time = {"t": None}
        def read_stream():
            for i, _ch in enumerate(streamer):
                if i == 0 and first_token_time["t"] is None:
                    torch.cuda.synchronize()
                    first_token_time["t"] = time.perf_counter()

        t = threading.Thread(target=read_stream)
        t.start()
        out_ids = model.generate(**gen_kwargs)
        save_prompt_and_completion(tok, input_ids.to("cpu"), out_ids.to("cpu"), run_idx=_)
        torch.cuda.synchronize()
        end_all = time.perf_counter()
        t.join()

        ttft = (first_token_time["t"] - start_prefill) if first_token_time["t"] else None
        total_time = end_all - start_prefill
        new_tokens = (out_ids.shape[-1] - input_ids.shape[-1])
        tokps = new_tokens / total_time if total_time > 0 else float("nan")

        ttft_list.append(ttft)
        gen_time_list.append(total_time)
        tokps_list.append(tokps)
        out_tokens_list.append(new_tokens)

    print("GPU PyTorch Benchmark")
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
