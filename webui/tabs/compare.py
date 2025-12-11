import streamlit as st
from typing import List

from .helpers import (
    BACKENDS,
    get_client,
    get_recommended_models,
    estimate,
    add_message,
    get_openai_model_ids,
    get_hf_model_ids,
)

def render():
    st.header("Compare models")
    st.markdown("Select up to 5 models (from any backend) to generate responses for the same prompt.")

    # Build options; prefer live lists for OpenAI and Hugging Face
    all_options: List[str] = []
    for b in BACKENDS:
        if b == "openai":
            models = get_openai_model_ids()
        elif b == "huggingface":
            models = get_hf_model_ids()
        else:
            try:
                models = get_recommended_models(b) or []
            except Exception:
                models = []
        for m in models:
            all_options.append(f"{b}::{m}")

    selected = st.multiselect(
        "Models (backend::model)",
        options=all_options,
        help="Format: backend::model",
        default=[],
    )
    if len(selected) > 5:
        st.error("Please select at most 5 models.")
        return

    prompt = st.text_area("Prompt", height=160, key="cmp_prompt")
    p_tokens = estimate(prompt)
    st.caption(f"Estimated prompt tokens: {p_tokens}")

    temp = st.slider("Temperature", 0.0, 2.0, 1.0, key="compare_temp")
    slider_options = [2**n for n in range(4, 16)]
    max_tokens = st.select_slider(
        "Max tokens",
        options=slider_options,
        value=1024,
        key="compare_max_tokens",
    )

    if st.button("Generate comparisons", key="cmp_generate"):
        if not selected:
            st.error("Select at least one model to compare.")
            return

        st.markdown(
            "<style>pre code{white-space:pre-wrap; word-wrap:break-word;} .stMarkdown{overflow-x:hidden;}</style>",
            unsafe_allow_html=True,
        )

        for sel in selected:
            try:
                backend, model = sel.split("::", 1)
            except Exception:
                backend, model = "local", sel

            client = get_client(backend, model)
            with st.expander(f"{backend} / {model}", expanded=True):
                placeholder = st.empty()
                out_acc = ""
                try:
                    for chunk in client.stream_generate(prompt or "", temperature=temp, max_tokens=max_tokens):
                        out_acc += chunk or ""
                        placeholder.code(out_acc)
                except Exception as e:
                    placeholder.error(f"Error: {e}")
                    out_acc = str(e)

                st.caption(f"Tokens≈{estimate(out_acc)} | Prompt≈{p_tokens}")

                add_message(
                    {
                        "role": "response",
                        "backend": backend,
                        "model": model,
                        "prompt": prompt,
                        "response": out_acc,
                        "prompt_tokens": p_tokens,
                        "response_tokens": estimate(out_acc),
                    }
                )