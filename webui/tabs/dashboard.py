import os
import pandas as pd
import altair as alt
import streamlit as st

def render():
    st.header("Benchmark Dashboard")
    st.markdown("Upload a `benchmark_results.csv` or place one at the project root and click 'Load file'.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
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
    agg = (
        df.groupby("client")
        .agg({"latency_s": ["mean", "median", "min", "max"], "resp_tokens": ["mean", "sum"]})
        .reset_index()
    )
    st.markdown("### Aggregated metrics by client")
    st.dataframe(agg)

    st.markdown("### Latency by client")
    chart = alt.Chart(df).mark_boxplot().encode(x="client", y="latency_s")
    st.altair_chart(chart, width='stretch')

    st.markdown("### Tokens vs Latency (scatter)")
    scatter = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(x="resp_tokens", y="latency_s", color="client", tooltip=["client", "latency_s", "resp_tokens"])
    )
    st.altair_chart(scatter, width='stretch')