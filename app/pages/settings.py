"""
Settings and Configuration
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

CSS = """
<style>
#MainMenu, footer, header { visibility: hidden; }
h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
h2, h3 { font-weight: 600 !important; }
hr { border-color: rgba(124, 111, 224, 0.2) !important; margin: 1.5rem 0; }
[data-testid="metric-container"] {
    background: rgba(124, 111, 224, 0.1);
    border: 1px solid rgba(124, 111, 224, 0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
</style>
"""

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-latest",
    "gemini-2.0-flash",
]


def _init_session_defaults() -> None:
    if "gemini_api_key" not in st.session_state:
        key = os.getenv("GOOGLE_API_KEY", "")
        if not key:
            try:
                key = st.secrets.get("GOOGLE_API_KEY", "")
            except Exception:
                pass
        st.session_state["gemini_api_key"] = key

    if "gemini_model" not in st.session_state:
        st.session_state["gemini_model"] = MODELS[0]


def main() -> None:
    _init_session_defaults()

    st.title("Settings")

    # ── Gemini ────────────────────────────────────────────────────────────────
    st.subheader("Gemini")
    col1, col2 = st.columns([2, 1])

    with col1:
        api_key = st.text_input(
            "Google API Key",
            value=st.session_state["gemini_api_key"],
            type="password",
            placeholder="AIza...",
            help="Used by the VLM Viewer to score video pairs.",
        )
        st.session_state["gemini_api_key"] = api_key

    with col2:
        model = st.selectbox(
            "Model",
            MODELS,
            index=MODELS.index(st.session_state["gemini_model"]) if st.session_state["gemini_model"] in MODELS else 0,
        )
        st.session_state["gemini_model"] = model

    if api_key:
        st.success("API key set — VLM Viewer is ready to score pairs.", icon="🔑")
    else:
        st.warning("No API key. The VLM Viewer Live tab will be disabled until you add one.")

    try:
        from src.preferences.vlm_gemini import PROMPT_VERSION
        st.caption(f"Prompt version: `{PROMPT_VERSION}`")
    except Exception:
        pass

    st.divider()

    # ── Storage ───────────────────────────────────────────────────────────────
    st.subheader("Storage")

    from src.storage.sheets import spreadsheet_id
    sid = spreadsheet_id()

    col3, col4 = st.columns(2)
    with col3:
        if sid:
            st.success("Google Sheets connected", icon="📊")
            st.caption(f"Spreadsheet ID: `{sid[:12]}...`")
        else:
            st.info("Local CSV files", icon="💾")
            st.caption("Add `spreadsheet_id` to `.streamlit/secrets.toml` to enable cloud storage.")

    with col4:
        try:
            hf_repo = st.secrets["huggingface"].get("video_repo_id", "")
        except Exception:
            hf_repo = ""
        if hf_repo:
            st.success(f"HuggingFace videos: `{hf_repo}`", icon="🤗")
        else:
            st.info("Videos served from local disk", icon="💽")


main()
