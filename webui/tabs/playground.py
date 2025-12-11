import time
from typing import Any, Dict, List

import streamlit as st

from .helpers import (
    BACKENDS,
    get_client,
    get_recommended_models,
    token_slider_options,
    credentials_status,
    estimate,
    filtered_openrouter_model_ids,
    provider_of,
    OPENROUTER_MODELS,
    add_message,
    render_conversation,
    get_openai_model_ids,
    get_hf_model_ids,
)


def render():
    st.header("Playground")
    col_settings, col_main = st.columns([1, 3])

    with col_settings:
        st.subheader("Settings")
        backend = st.selectbox("Backend", BACKENDS)

        # Model selection per backend
        if backend == "openrouter":
            st.markdown("#### OpenRouter filters")
            free_only = st.checkbox("Free models only", value=True, key="pl_free_only")
            providers = sorted(set(filter(None, (provider_of(m) or "unknown" for m in OPENROUTER_MODELS))))
            provider_sel = st.selectbox("Provider", options=["All"] + providers, key="pl_provider")
            options = filtered_openrouter_model_ids(free_only, provider_sel)
            if not options:
                options = get_recommended_models("openrouter")
            model = st.selectbox("OpenRouter model", options=options, key="pl_model")

            try:
                selected = next((m for m in OPENROUTER_MODELS if str(m.get("id") or m.get("model")) == model), None)
            except Exception:
                selected = None
            with st.expander("Model info", expanded=False):
                if selected:
                    st.markdown(f"**ID:** {selected.get('id')}")
                    st.markdown(f"**Provider:** {provider_of(selected) or 'unknown'}")
                    # desc = model.get('description') or selected.get('summary') or selected.get('label')
                    desc = selected.get('description')
                    if desc:
                        st.write(desc)
                    st.markdown("---")
                    st.write(selected)
                else:
                    st.write("No model metadata available.")
                  
        elif backend == "openai":
            # Live OpenAI models (cached) with static fallback
            options = get_openai_model_ids()
            if not options:
                st.warning("No OpenAI models available. Check OPENAI_API_KEY or connectivity.")
                options = get_recommended_models("openai")
            model = st.selectbox("OpenAI model", options=options, key="pl_model")
        
        elif backend == "huggingface":
            # Live Hugging Face models (cached via Hub API) with static fallback
            options = get_hf_model_ids()
            if not options:
                st.warning("No Hugging Face models available. Check HF_API_TOKEN or connectivity.")
                options = get_recommended_models("huggingface")
            model = st.selectbox("Hugging Face model", options=options, key="pl_model")
            st.caption("Uses Hugging Face Inference Router (router.huggingface.co). Some models may not support streaming.")
        
        else:
            # Local/ollama or transformers
            local_examples = ["distilgpt2", "gpt2", "EleutherAI/gpt-neo-125M"]
            options = list(dict.fromkeys(get_recommended_models("local") + local_examples))
            model = st.selectbox("Local model", options=options, key="pl_model")
            st.caption("Local models use transformers or Ollama. First run may download weights.")

        st.markdown("---")
        template_keys = list(st.session_state["templates"].keys()) if "templates" in st.session_state else []
        template = st.selectbox("Template", options=template_keys, key="pl_template") if template_keys else None
        tpl_text = st.session_state["templates"][template] if template else ""

        st.markdown("---")
        temp = st.slider("Temperature", 0.0, 2.0, 1.0, key="pl_temp")
        max_tokens = st.select_slider(
            "Max tokens",
            options=token_slider_options(backend, model),
            value=1024,
            key="pl_max_tokens",
        )

        st.markdown("---")
        st.markdown("Credentials")
        creds = credentials_status()
        st.write({k: "✅" if v else "❌" for k, v in creds.items()})

    with col_main:
        st.subheader("Prompt")
        prompt = st.text_area("Prompt", value=tpl_text, height=160, key="pl_prompt")
        p_tokens = estimate(prompt)
        st.caption(f"Estimated prompt tokens: {p_tokens}")

        c1, c2, _ = st.columns([1, 1, 1])
        gen = c1.button("Generate", key="pl_generate")
        clear = c2.button("Clear", key="pl_clear")

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
                        text_accum += chunk or ""
                        placeholder.text(text_accum)
                except Exception as e:
                    st.error(f"Error during generation: {e}")
                    text_accum = str(e)
            entry: Dict[str, Any] = {
                "role": "response",
                "backend": backend,
                "model": model,
                "prompt": prompt,
                "response": text_accum,
                "prompt_tokens": p_tokens,
                "response_tokens": estimate(text_accum),
                "latency_s": round(time.time() - t0, 3),
            }
            add_message(entry)

        st.markdown("---")
        st.subheader("Conversation")
        render_conversation()