import streamlit as st
from webui.tabs import playground, compare, models, dashboard
from webui.tabs.helpers import ensure_session

def main():
    st.set_page_config(page_title="LLM Wrappers", layout="wide")
    ensure_session()
    tabs = st.tabs(["Playground", "Compare", "Models", "Dashboard"])
    with tabs[0]:
        playground.render()
    with tabs[1]:
        compare.render()
    with tabs[2]:
        models.render()
    with tabs[3]:
        dashboard.render()

if __name__ == "__main__":
    main()