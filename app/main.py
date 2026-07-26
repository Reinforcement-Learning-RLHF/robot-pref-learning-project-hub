"""
App entry point — defines navigation order, icons, and page titles.
All page content lives in app/pages/.
"""

import streamlit as st

st.set_page_config(
    page_title="Robot Pref Learning",
    page_icon="🤖",
    layout="wide",
)

# Global CSS injected once here, applies to every page
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
h2, h3 { font-weight: 600 !important; }
h4 { color: #9D98E8 !important; font-weight: 600 !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #7C6FE0, #5A4FCF); border-radius: 4px;
}
[data-testid="metric-container"] {
    background: rgba(124, 111, 224, 0.1);
    border: 1px solid rgba(124, 111, 224, 0.2);
    border-radius: 10px; padding: 0.75rem 1rem;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #7C6FE0 0%, #5A4FCF 100%);
    border: none; border-radius: 8px; font-weight: 600;
    letter-spacing: 0.3px; transition: all 0.15s ease;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8D82E8 0%, #6A5FDF 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(124, 111, 224, 0.35);
}
[data-testid="stButton"] > button[kind="secondary"] { border-radius: 8px; font-weight: 500; }
hr { border-color: rgba(124, 111, 224, 0.2) !important; margin: 1.5rem 0; }
.task-badge {
    display: inline-block; padding: 0.25rem 0.9rem; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase;
    background: rgba(124, 111, 224, 0.15);
    border: 1px solid rgba(124, 111, 224, 0.35); color: #9D98E8;
    margin-bottom: 0.25rem;
}
.score-badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin-left: 0.5rem;
}
.score-b-wins { background: rgba(239,83,80,0.15);   color: #EF5350; border: 1px solid rgba(239,83,80,0.3); }
.score-a-wins { background: rgba(102,187,106,0.15); color: #66BB6A; border: 1px solid rgba(102,187,106,0.3); }
.score-tie    { background: rgba(255,167,38,0.15);  color: #FFA726; border: 1px solid rgba(255,167,38,0.3); }
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/home.py",          title="Home",     icon="🏠", default=True),
    st.Page("pages/scripted.py",      title="Scripted", icon="🔧"),
    st.Page("pages/human.py",         title="Human",    icon="🏷️"),
    st.Page("pages/vlm_viewer.py",    title="VLM",      icon="🤖"),
    st.Page("pages/hdf5_explorer.py", title="HDF5",     icon="🗂️"),
    st.Page("pages/settings.py",      title="Settings", icon="⚙️"),
])
pg.run()
