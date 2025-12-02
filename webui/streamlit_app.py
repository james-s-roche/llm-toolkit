"""Streamlit UI with streaming and results dashboard for llm-wrappers.

Features:
- Playground: streaming outputs, templates, conversation history, compare mode
- Dashboard: visualize benchmark CSV (latency, tokens) with pandas/altair
"""
import json
import time
import os
from typing import Any, Dict, List, Optional

import streamlit as st

import pandas as pd
import altair as alt

from llm_wrappers.models_registry import RECOMMENDED
from llm_wrappers.token_counter import estimate_tokens
from llm_wrappers.openai_client import OpenAIClient
from llm_wrappers.hf_client import HFClient
from llm_wrappers.local_client import LocalClient
from llm_wrappers.ollama_client import OllamaClient
from llm_wrappers.openrouter_client import OpenRouterClient
from llm_wrappers.secrets import get_secret


from llm_wrappers.openrouter_utils import fetch_openrouter_models, sort_models, is_free_model, provider_of

# Prefer OpenRouter as the default backend
BACKENDS = ["openrouter", "openai", "huggingface", "local"]

# OpenRouter API endpoint used for inference calls (model id is passed separately)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# Fetch OpenRouter models at app start (falls back to cached copy on network failure)
OPENROUTER_MODELS = []
try:
    OPENROUTER_MODELS = sort_models(fetch_openrouter_models())
except Exception:
    OPENROUTER_MODELS = []


def get_recommended_models(backend: str) -> List[str]:
    if backend == "openai":
        return [m["id"] for m in RECOMMENDED.get("openai", [])]
    if backend == "huggingface":
        return [m["id"] for m in RECOMMENDED.get("huggingface", [])]
    if backend == "openrouter":
        # Use dynamic OpenRouter model list if available, else registry
        if OPENROUTER_MODELS:
            return [str(m.get("id") or m.get("model")) for m in OPENROUTER_MODELS]
        return [m["id"] for m in RECOMMENDED.get("openrouter", [])]
    return [m["id"] for m in RECOMMENDED.get("local", [])]


def _get_max_tokens_from_openrouter_model(m: Dict[str, Any]) -> Optional[int]:
    # heuristics: check common fields used by providers
    for k in ("max_tokens", "context", "context_window", "max_context", "max_input_tokens", "input_context"):
        v = m.get(k)
        try:
            if isinstance(v, (int, float)) and int(v) > 0:
                return int(v)
            if isinstance(v, str) and v.isdigit():
                return int(v)
        except Exception:
            continue
    # sometimes price or tags contain numbers — skip those
    return None


def get_model_max_tokens(backend: str, model: str) -> int:
    """Return a recommended max token limit for the given backend+model.

    Uses OpenRouter model metadata when available; otherwise falls back to
    sensible defaults per backend.
    """
    backend = (backend or "").lower()
    # try OpenRouter metadata
    if backend == "openrouter":
        try:
            sel = next((m for m in OPENROUTER_MODELS if str(m.get("id") or m.get("model")) == str(model)), None)
            if sel:
                v = _get_max_tokens_from_openrouter_model(sel)
                if v:
                    return max(256, v)
        except Exception:
            pass

    # backend-specific sensible defaults
    defaults = {
        "openrouter": 32768,
        "openai": 32768,
        "huggingface": 8192,
        "local": 2048,
    }
    return defaults.get(backend, 2048)


def power_of_two_options(min_val: int = 16, max_val: int = 1024) -> List[int]:
    opts = []
    v = max(16, min_val)
    while v <= max_val:
        opts.append(v)
        v *= 2
    # ensure min_val present
    if opts and opts[0] > min_val:
        opts.insert(0, min_val)
    return opts


def get_client(backend: str, model: str):
    backend = backend.lower()
    if backend == "openai":
        api_key = get_secret("OPENAI_API_KEY")
        return OpenAIClient(model=model, api_key=api_key)
    if backend == "huggingface":
        hf_token = get_secret("HF_API_TOKEN")
        return HFClient(model=model, hf_api_token=hf_token)
    if backend == "local":
        # support Ollama model ids using prefix 'ollama:' or 'ollama/'
        if isinstance(model, str) and (model.startswith("ollama:") or model.startswith("ollama/")):
            # extract downstream model name after the prefix
            cleaned = model.split(":", 1)[1] if ":" in model else model.split("/", 1)[1]
            # allow configuring Ollama host via env var OLLAMA_HOST
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            return OllamaClient(model=cleaned, host=host)
        return LocalClient(model=model)
    if backend == "openrouter":
        api_key = get_secret("OPENROUTER_API_KEY")
        # pass the proper API endpoint and set the selected model id via `model`
        return OpenRouterClient(api_url=OPENROUTER_API_URL, api_key=api_key, model=model)
    raise ValueError("Unknown backend")


def ensure_session():
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = []
    if "templates" not in st.session_state:
        st.session_state["templates"] = {
            "Greeting": "Write a friendly greeting in one sentence.",
            "ELI5": "Explain {topic} like I'm 5.",
            "Summarize": "Summarize the following text:\n{input}",
        }


def add_message(entry: Dict[str, Any]):
    st.session_state["conversations"].append(entry)


def render_conversation():
    for i, msg in enumerate(reversed(st.session_state["conversations"])):
        with st.expander(f"[{msg.get('backend','')}/{msg.get('model','')}] {msg.get('role','resp')}"):
            st.markdown("**Prompt**")
            st.write(msg.get("prompt", ""))
            st.markdown("**Response**")
            st.write(msg.get("response", ""))
            st.markdown("**Meta**")
            st.write({k: v for k, v in msg.items() if k not in ("prompt", "response")})


def playground_tab():
    st.header("Playground")
    st.sidebar.title("Settings")
    backend = st.sidebar.selectbox("Backend", BACKENDS)

    recommended = get_recommended_models(backend)
    if backend == "openai":
        model = st.sidebar.selectbox("Model", options=(recommended + ["gpt-3.5-turbo", "gpt-4o-mini"]))
    elif backend == "openrouter":
        # Provide filters for OpenRouter: free/paid and provider
        st.sidebar.markdown("#### OpenRouter filters")
        free_only = st.sidebar.checkbox("Free models only", value=True)
        # collect providers
        providers = [provider_of(m) or "unknown" for m in OPENROUTER_MODELS]
        providers = sorted(set([p for p in providers if p]))
        provider_sel = st.sidebar.selectbox("Provider", options=["All"] + providers)

        # build options from OPENROUTER_MODELS (sorted so free come first)
        options = []
        for m in OPENROUTER_MODELS:
            if free_only and not is_free_model(m):
                continue
            prov = provider_of(m) or "unknown"
            if provider_sel != "All" and prov != provider_sel:
                continue
            options.append(str(m.get("id") or m.get("model") or m.get("name") or ""))

        if not options:
            options = recommended + ["https://api.openrouter.example/model"]

        model = st.sidebar.selectbox("OpenRouter model", options=options)

        # show selected model info
        try:
            selected = next((m for m in OPENROUTER_MODELS if str(m.get("id") or m.get("model")) == model), None)
        except Exception:
            selected = None
        with st.sidebar.expander("Model info", expanded=False):
            if selected:
                st.markdown(f"**ID:** {selected.get('id')}")
                st.markdown(f"**Provider:** {provider_of(selected) or 'unknown'}")
                desc = selected.get('description') or selected.get('summary') or selected.get('label')
                if desc:
                    st.write(desc)
                st.markdown("---")
                st.write(selected)
            else:
                st.write("No model metadata available.")
    elif backend == "local":
        # include smaller Hugging Face-hosted local model examples (no standalone app required)
        # Note: LocalClient uses `transformers` and will download model weights from the Hugging Face Hub
        # on first use. These are smaller models that are reasonable for local CPU runs.
        local_examples = ["distilgpt2", "gpt2", "EleutherAI/gpt-neo-125M", "ollama:llama2"]
        model = st.sidebar.selectbox("Local model", options=(recommended + local_examples))
        st.sidebar.caption("Local models use the Hugging Face `transformers` library and will download weights the first time they are used; ensure you have disk space and transformers installed.")
    else:
        model = st.sidebar.selectbox("Model", options=recommended or ["gpt2"]) 

    st.sidebar.markdown("### Credentials")
    st.sidebar.write(f"OPENAI_API_KEY: {'✅' if get_secret('OPENAI_API_KEY') else '❌'}")
    st.sidebar.write(f"HF_API_TOKEN: {'✅' if get_secret('HF_API_TOKEN') else '❌'}")
    st.sidebar.write(f"OPENROUTER_API_KEY: {'✅' if get_secret('OPENROUTER_API_KEY') else '❌'}")

    st.sidebar.markdown("---")
    temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.sidebar.slider("Max tokens", 16, 1024, 256)

    st.sidebar.markdown("---")
    template = st.sidebar.selectbox("Template", options=list(st.session_state["templates"].keys()))
    tpl_text = st.session_state["templates"][template]

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.subheader("Prompt")
        prompt = st.text_area("Prompt", value=tpl_text, height=160)
        p_tokens = estimate_tokens(prompt)
        st.caption(f"Estimated prompt tokens: {p_tokens}")

        c1, c2, c3 = st.columns([1, 1, 1])
        gen = c1.button("Generate")
        clear = c2.button("Clear")

        if clear:
            st.session_state["conversations"] = []

        if gen:
            client = get_client(backend, model)
            placeholder = st.empty()
            text_accum = ""
            t0 = time.time()
            with st.spinner("Streaming response..."):
                try:
                    for chunk in client.stream_generate(prompt, temperature=temp, max_tokens=max_tokens):
                        text_accum += chunk
                        # update UI progressively
                        placeholder.markdown("""**Response (streaming)**
```
{0}
```""".format(text_accum))
                except Exception as e:
                    st.error(f"Error during generation: {e}")
                    text_accum = str(e)
            t1 = time.time()
            r_tokens = estimate_tokens(text_accum or "")
            entry = {
                "role": "response",
                "backend": backend,
                "model": model,
                "prompt": prompt,
                "response": text_accum,
                "prompt_tokens": p_tokens,
                "response_tokens": r_tokens,
                "latency_s": round(t1 - t0, 3),
            }
            add_message(entry)

        # Compare feature moved to its own tab

        st.markdown("---")
        st.subheader("Conversation")
        render_conversation()

    with col_side:
        st.markdown("### Controls")
        if st.session_state["conversations"]:
            last = st.session_state["conversations"][-1]
            st.write({"backend": last.get("backend"), "model": last.get("model"), "latency_s": last.get("latency_s"), "resp_tokens": last.get("response_tokens")})
        else:
            st.write("No calls yet")
        st.markdown("---")
        st.markdown("### Export")
        if st.session_state["conversations"]:
            data = json.dumps(st.session_state["conversations"], indent=2)
            st.download_button("Download JSON", data=data, file_name="conversation.json", mime="application/json")


def dashboard_tab():
    st.header("Benchmark Dashboard")
    st.markdown("Upload a `benchmark_results.csv` or place one at the project root and click 'Load file'.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"] )
    load_button = st.button("Load benchmark_results.csv from repo")

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
    elif load_button:
        path = os.path.join(os.getcwd(), "benchmark_results.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")
        else:
            st.error("benchmark_results.csv not found in project root")

    if df is None:
        st.info("No benchmark data loaded yet.")
        return

    st.write(df.head())
    # Basic aggregations
    agg = df.groupby("client").agg({"latency_s": ["mean", "median", "min", "max"], "resp_tokens": ["mean", "sum"]}).reset_index()
    st.markdown("### Aggregated metrics by client")
    st.dataframe(agg)

    # Charts
    st.markdown("### Latency by client")
    chart = alt.Chart(df).mark_boxplot().encode(x="client", y="latency_s")
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Tokens vs Latency (scatter)")
    scatter = alt.Chart(df).mark_circle(size=60).encode(x="resp_tokens", y="latency_s", color="client", tooltip=["client", "latency_s", "resp_tokens"]) 
    st.altair_chart(scatter, use_container_width=True)


def compare_tab():
    st.header("Compare models")
    st.markdown("Select up to 5 models (from any backend) to generate responses for the same prompt and compare outputs side-by-side.")

    # build a combined list of backend::model options
    all_options = []
    for b in BACKENDS:
        try:
            rec = get_recommended_models(b) or []
        except Exception:
            rec = []
        for m in rec:
            all_options.append(f"{b}::{m}")

    selected = st.multiselect("Models (backend::model)", options=all_options, help="Format: backend::model", default=[])
    if len(selected) > 5:
        st.error("Please select at most 5 models.")
        return

    prompt = st.text_area("Prompt", height=160)
    p_tokens = estimate_tokens(prompt or "")
    st.caption(f"Estimated prompt tokens: {p_tokens}")

    temp = st.slider("Temperature", 0.0, 1.0, 0.2, key="compare_temp")
    max_tokens = st.slider("Max tokens", 16, 1024, 256, key="compare_max_tokens")

    if st.button("Generate comparisons"):
        if not selected:
            st.error("Select at least one model to compare.")
            return

        cols = st.columns(len(selected))
        results = ["" for _ in selected]

        # stream into each column sequentially
        for i, sel in enumerate(selected):
            try:
                backend, model = sel.split("::", 1)
            except Exception:
                backend, model = "local", sel

            client = get_client(backend, model)
            col = cols[i]
            with col:
                st.markdown(f"**{backend} / {model}**")
                placeholder = st.empty()
                out_acc = ""
                try:
                    for chunk in client.stream_generate(prompt or "", temperature=temp, max_tokens=max_tokens):
                        out_acc += chunk
                        placeholder.code(out_acc, language=None)
                except Exception as e:
                    placeholder.error(f"Error: {e}")
                    out_acc = str(e)

                # store entry for each model
                add_message({
                    "role": "response",
                    "backend": backend,
                    "model": model,
                    "prompt": prompt,
                    "response": out_acc,
                    "prompt_tokens": p_tokens,
                    "response_tokens": estimate_tokens(out_acc or ""),
                })


def models_tab():
    st.header("OpenRouter Models Catalog")
    st.markdown("Browse the OpenRouter model catalog (fetched live). Use the filters to narrow results and click Refresh to re-fetch the catalog.")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("### Filters")
        free_only = st.checkbox("Free models only", value=True)
        providers = [provider_of(m) or "unknown" for m in OPENROUTER_MODELS]
        providers = sorted(set([p for p in providers if p]))
        provider_sel = st.selectbox("Provider", options=["All"] + providers)
        refresh = st.button("Refresh catalog")

    if refresh:
        with st.spinner("Fetching OpenRouter models..."):
            try:
                new = sort_models(fetch_openrouter_models())
                # update module-level cache for the running app without using 'global'
                import sys
                mod = sys.modules[__name__]
                setattr(mod, "OPENROUTER_MODELS", new)
                st.success(f"Fetched {len(new)} models")
            except Exception as e:
                st.error(f"Failed to fetch models: {e}")

    # Search/filter input
    query = col1.text_input("Search models (id, name, description, tags)")

    # Build filtered list
    def matches(m):
        if free_only and not is_free_model(m):
            return False
        if provider_sel != "All" and (provider_of(m) or "unknown") != provider_sel:
            return False
        if query:
            q = query.lower()
            vals = [str(m.get("id") or ""), str(m.get("model") or ""), str(m.get("name") or ""), str(m.get("description") or "")]
            tags = m.get("tags") or []
            vals.extend([str(t) for t in tags])
            if not any(q in v.lower() for v in vals):
                return False
        return True

    filtered = [m for m in OPENROUTER_MODELS if matches(m)] if OPENROUTER_MODELS else []

    st.markdown(f"**{len(filtered)} models found**")

    show_json = st.checkbox("Show refashioned JSON", value=False)
    if show_json:
        st.json(filtered)

    # Paginate display: show up to 50 models with expanders
    for m in filtered[:200]:
        title = m.get("id") or m.get("model") or m.get("name") or "<unknown>"
        with st.expander(title):
            st.markdown(f"**Provider:** {provider_of(m) or 'unknown'}")
            desc = m.get("description") or m.get("summary") or m.get("label")
            if desc:
                st.write(desc)
            st.markdown("**Metadata**")
            # show a trimmed dict
            trimmed = {k: v for k, v in m.items() if k in ("id", "model", "name", "description", "tags", "provider", "price")}
            st.write(trimmed)
            # link to model or provider if present
            if m.get("url"):
                st.markdown(f"[Model page]({m.get('url')})")
            if provider_of(m):
                st.markdown(f"**Provider:** {provider_of(m)}")

    if filtered:
        # allow download of filtered results
        data = json.dumps(filtered, indent=2)
        st.download_button("Download filtered models JSON", data=data, file_name="openrouter_models.json", mime="application/json")


def main():
    st.set_page_config(page_title="LLM Wrappers", layout="wide")
    ensure_session()
    tabs = st.tabs(["Playground", "Compare", "Models", "Dashboard"])
    with tabs[0]:
        playground_tab()
    with tabs[1]:
        compare_tab()
    with tabs[2]:
        models_tab()
    with tabs[3]:
        dashboard_tab()


if __name__ == "__main__":
    main()

