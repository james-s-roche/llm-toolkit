import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from llm_wrappers.secrets import get_secret
from llm_wrappers.openrouter_utils import fetch_openrouter_models
from llm_wrappers.openai_client import OpenAIClient
from llm_wrappers.hf_client import HFClient


# Cache fetches for a short period; allow refresh button to clear
@st.cache_data(show_spinner=False, ttl=300)
def _fetch_openrouter_models_cached() -> List[Dict[str, Any]]:
    try:
        return fetch_openrouter_models()
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_openai_models_cached(api_key: Optional[str]) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    try:
        client = OpenAIClient(api_key=api_key)
        # Convert to plain dicts so we can json_normalize easily
        return [ {"provider":"openai", **vars(m)} for m in client.models() ]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_hf_models_cached(token: Optional[str], limit: int = 300) -> List[Dict[str, Any]]:
    try:
        client = HFClient(hf_api_token=token)
        models = client.models(pipeline_tag="text-generation", limit=limit)
        return [ {"provider":"huggingface", **vars(m)} for m in models ]
    except Exception:
        return []


def _normalize(models: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize a list of dicts with pandas.json_normalize without guessing key names.
    """
    if not models:
        return pd.DataFrame()
    try:
        df = pd.json_normalize(models, max_level=3)
    except Exception:
        # Last-resort conversion
        df = pd.DataFrame(models)
    return df


def render():
    st.header("Models")

    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        include_openrouter = st.checkbox("OpenRouter", value=True)
    with c2:
        include_openai = st.checkbox("OpenAI", value=True)
    with c3:
        include_hf = st.checkbox("Hugging Face", value=True)
    with c4:
        if st.button("Refresh"):
            _fetch_openrouter_models_cached.clear()
            _fetch_openai_models_cached.clear()
            _fetch_hf_models_cached.clear()
            st.rerun()

    # Fetch
    combined_json: List[Dict[str, Any]] = []

    if include_openrouter:
        orm = _fetch_openrouter_models_cached()
        # Tag provider and keep raw structure
        for m in orm:
            combined_json.append({"provider": "openrouter", **m})

    if include_openai:
        openai_key = get_secret("OPENAI_API_KEY")
        oai = _fetch_openai_models_cached(openai_key)
        combined_json.extend(oai)

    if include_hf:
        hf_token = get_secret("HF_API_TOKEN")
        hfm = _fetch_hf_models_cached(hf_token)
        combined_json.extend(hfm)

    # Summary
    st.caption(f"Total models loaded: {len(combined_json)}")

    # Normalize
    df = _normalize(combined_json)

    if df.empty:
        st.info("No models to display. Check provider checkboxes and credentials.")
        return

    # Optional filtering controls
    with st.expander("Filters and columns", expanded=False):
        # Text search over all columns (string contains)
        search = st.text_input("Search (substring across all visible columns)")
        # Column selection
        all_cols = list(df.columns)
        # default_cols = [c for c in all_cols if c in ("provider", "id", "name", "created", "owned_by", "likes")]
        default_cols = [c for c in all_cols if c in 
                        ("provider", "id", "name", "context_length", "architecture.modality", 
                         "created", "supported_parameters", "description")]
        selected_cols = st.multiselect("Columns to display", options=all_cols, default=default_cols or all_cols)

    # Apply filters
    df_view = df
    if selected_cols:
        # Keep selected columns that exist
        keep = [c for c in selected_cols if c in df_view.columns]
        if keep:
            df_view = df_view[keep]

    if search:
        # Build mask: any column as string contains the search substring
        s = search.lower()
        mask = pd.Series([False] * len(df_view))
        for col in df_view.columns:
            try:
                col_str = df_view[col].astype(str).str.lower()
                mask = mask | col_str.str.contains(s, na=False)
            except Exception:
                continue
        df_view = df_view[mask]

    st.dataframe(df_view, 
                 width='stretch',
                 column_config={'created': st.column_config.DatetimeColumn(
                     'Created', format="YYYY-MM-DD",
                 )})

    # Downloads and raw JSON
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (all columns)", 
                           data=csv_bytes, 
                           file_name="models_all.csv", 
                           mime="text/csv")
    with col_dl2:
        st.download_button("Download JSON (raw)", 
                           data=json.dumps(combined_json, indent=2).encode("utf-8"), 
                           file_name="models_raw.json", 
                           mime="application/json")

    with st.expander("Raw JSON", expanded=False):
        st.json(combined_json)