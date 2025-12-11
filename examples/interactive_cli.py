"""Simple interactive CLI to choose a backend and send prompts."""
import argparse
import os

from llm_wrappers.base import BaseLLM
from llm_wrappers.openai_client import OpenAIClient
from llm_wrappers.hf_client import HFClient
from llm_wrappers.local_client import LocalClient
from llm_wrappers.secrets import set_env_from_secret, get_secret


def pick_client(name: str, model: str):
    name = name.lower()
    if name == "openai":
        # Prefer environment, then .env, then macOS Keychain
        key = get_secret("OPENAI_API_KEY")
        return OpenAIClient(api_key=key, model=model)
    if name == "hf":
        token = get_secret("HF_API_TOKEN")
        return HFClient(model=model, hf_api_token=token)
    if name == "local":
        return LocalClient(model=model)
    raise ValueError("Unknown backend: %s" % name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["openai", "hf", "local"], default="openai")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    model = args.model
    if model is None:
        # Sensible defaults
        if args.backend == "openai":
            model = "gpt-3.5-turbo"
        elif args.backend == "hf":
            model = "gpt2"
        else:
            model = "gpt2"

    client = pick_client(args.backend, model)

    print(f"Using backend={args.backend} model={model}")
    print("Enter a prompt (empty to quit):")
    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:
            break
        if not prompt:
            break
        try:
            out = client.generate(prompt)
            # Print result neatly
            if isinstance(out, dict) and "text" in out:
                print("\n== Generated ==")
                print(out["text"].strip())
                print("==============\n")
            else:
                print(out)
        except Exception as e:
            print("Error from client:", e)


if __name__ == "__main__":
    main()
