"""Simple benchmarking harness for LLM clients.

Usage: run this script from the repo root. It will import your chosen client
and run a few requests, recording latency and token estimates.
"""
import time
import csv
from typing import Callable, Dict

from llm_wrappers.token_counter import estimate_tokens
from llm_wrappers.secrets import get_secret


def benchmark_client(name: str, client_factory: Callable[[], object], prompts, repeat=3):
    rows = []
    client = client_factory()
    for prompt in prompts:
        for i in range(repeat):
            t0 = time.time()
            out = client.generate(prompt)
            t1 = time.time()
            text = out.get("text") if isinstance(out, dict) else str(out)
            tok = estimate_tokens(text or "")
            rows.append({
                "client": name,
                "prompt_len": len(prompt),
                "resp_tokens": tok,
                "latency_s": round(t1 - t0, 4),
            })
    return rows


def save_csv(rows, path="benchmark_results.csv"):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Example usage: change factories to suit your env.
    from llm_wrappers.openai_client import OpenAIClient
    from llm_wrappers.hf_client import HFClient

    prompts = [
        "Write a two-sentence summary of the benefits of caching.",
        "Explain gradient descent in simple terms.",
    ]

    rows = []
    def oa_factory():
        key = get_secret("OPENAI_API_KEY")
        return OpenAIClient(api_key=key)

    def hf_factory():
        token = get_secret("HF_API_TOKEN")
        return HFClient(hf_api_token=token)

    try:
        rows += benchmark_client("openai", oa_factory, prompts)
    except Exception as e:
        print("OpenAI benchmark skipped:", e)
    try:
        rows += benchmark_client("huggingface", hf_factory, prompts)
    except Exception as e:
        print("HF benchmark skipped:", e)

    save_csv(rows)
    print("Wrote benchmark_results.csv with", len(rows), "rows")
