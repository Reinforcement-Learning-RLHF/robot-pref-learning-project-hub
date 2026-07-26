"""
VLM Preference Viewer

Browse tab  -- explore stored Gemini labels
Live tab    -- pick two rollouts, call Gemini now, optionally save the result
"""

import csv
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

VLM_CSV = ROOT / "data" / "preferences" / "vlm_pairs.csv"
REGISTRY_CSV = ROOT / "data" / "rollout_registry.csv"
VLM_SHEET = "vlm_pairs"

FIELDNAMES = [
    "traj_A", "traj_B", "label", "evaluator_type", "timestamp", "model_version", "prompt_version",
    "score_ab", "score_ba_flipped", "comment_ab", "comment_ba", "consistent",
]

CSS = """
<style>
#MainMenu, footer, header { visibility: hidden; }
h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
h2, h3 { font-weight: 600 !important; }
h4 { color: #9D98E8 !important; font-weight: 600 !important; }
[data-testid="metric-container"] {
    background: rgba(124, 111, 224, 0.1);
    border: 1px solid rgba(124, 111, 224, 0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="stSidebar"] h2 {
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7C6FE0 !important;
    margin-bottom: 0.75rem;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #7C6FE0 0%, #5A4FCF 100%);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.15s ease;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8D82E8 0%, #6A5FDF 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(124, 111, 224, 0.35);
}
hr { border-color: rgba(124, 111, 224, 0.2) !important; margin: 1.5rem 0; }
[data-testid="stVideo"] video { max-width: 480px; width: 100%; display: block; margin: 0 auto; }
.score-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
.score-b-wins { background: rgba(239,83,80,0.15);   color: #EF5350; border: 1px solid rgba(239,83,80,0.3); }
.score-a-wins { background: rgba(102,187,106,0.15); color: #66BB6A; border: 1px solid rgba(102,187,106,0.3); }
.score-tie    { background: rgba(255,167,38,0.15);  color: #FFA726; border: 1px solid rgba(255,167,38,0.3); }
</style>
"""


# ── Storage helpers ───────────────────────────────────────────────────────────

def _sheets_id() -> str | None:
    from src.storage.sheets import spreadsheet_id
    return spreadsheet_id()


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets["huggingface"].get("hf_token")
    except Exception:
        return None


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_registry() -> pd.DataFrame:
    df = pd.read_csv(REGISTRY_CSV, encoding="utf-8-sig")
    df["success_label"] = df["success_label"].astype(str).str.lower() == "true"
    if "video_url" not in df.columns:
        df["video_url"] = ""
    return df


@st.cache_data
def load_video_bytes(path_str: str, video_url: str | None = None) -> bytes | None:
    """Load video as H.264 bytes. Tries local path first, then video_url (HF Hub)."""
    path = Path(path_str)

    if path.exists():
        src_path = str(path)
        is_remote = False
    elif video_url:
        headers = {}
        token = _hf_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            r = requests.get(video_url, headers=headers, timeout=60)
            r.raise_for_status()
        except Exception as exc:
            st.warning(f"Could not fetch video: {exc}")
            return None
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(r.content)
            src_path = tmp.name
        is_remote = True
    else:
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_tmp:
        out_path = out_tmp.name

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", "-an", out_path],
        capture_output=True,
    )
    if is_remote:
        Path(src_path).unlink(missing_ok=True)

    if result.returncode == 0:
        data = Path(out_path).read_bytes()
        Path(out_path).unlink(missing_ok=True)
        return data

    Path(out_path).unlink(missing_ok=True)
    if not is_remote and path.exists():
        return path.read_bytes()
    return None


@st.cache_data(ttl=30)
def load_vlm_pairs() -> pd.DataFrame | None:
    sid = _sheets_id()
    if sid:
        from src.storage.sheets import read_rows
        rows = read_rows(sid, VLM_SHEET)
        return pd.DataFrame(rows) if rows else None
    if not VLM_CSV.exists() or VLM_CSV.stat().st_size == 0:
        return None
    df = pd.read_csv(VLM_CSV)
    return df if len(df) > 0 else None


def save_pair(row: dict) -> None:
    sid = _sheets_id()
    if sid:
        from src.storage.sheets import append_row
        append_row(sid, VLM_SHEET, row, FIELDNAMES)
    else:
        VLM_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not VLM_CSV.exists() or VLM_CSV.stat().st_size == 0
        with VLM_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    load_vlm_pairs.clear()


# ── Shared UI components ──────────────────────────────────────────────────────

def render_video_col(col, label: str, rollout_id: str, registry: pd.DataFrame) -> None:
    with col:
        with st.container(border=True):
            st.markdown(f"#### Rollout {label}")
            st.caption(f"`{rollout_id}`")
            rows = registry[registry["rollout_id"] == rollout_id]
            if rows.empty:
                st.warning(f"`{rollout_id}` not in registry.")
            else:
                r = rows.iloc[0]
                video_path = ROOT / r["video_path"]
                _raw_url = r.get("video_url")
                video_url = str(_raw_url).strip() if _raw_url and str(_raw_url) not in ("nan", "", "None") else None
                data = load_video_bytes(str(video_path), video_url)
                if data:
                    st.video(data, format="video/mp4")
                else:
                    st.warning(f"Video not found: `{r['video_path']}`")


def _score_label(score: float) -> str:
    if score > 0.55:
        return f'<span class="score-badge score-a-wins">A wins ({score:.2f})</span>'
    if score < 0.45:
        return f'<span class="score-badge score-b-wins">B wins ({score:.2f})</span>'
    return f'<span class="score-badge score-tie">Tie ({score:.2f})</span>'


def render_scores(label, score_ab=None, score_ba_flipped=None, comment_ab="", comment_ba="", consistent=None) -> None:
    st.markdown(f"**Gemini verdict** {_score_label(label)}", unsafe_allow_html=True)
    st.write("")

    m1, m2, m3 = st.columns(3)
    m1.metric("Combined score", f"{label:.2f}", help="1.0 = A better  |  0.0 = B better  |  0.5 = tie")
    if score_ab is not None:
        m2.metric("AB score (raw)", f"{float(score_ab):.2f}")
    if score_ba_flipped is not None:
        m3.metric("BA score (flipped)", f"{float(score_ba_flipped):.2f}")
    st.caption("Score guide: 0.0 = B clearly better · 0.25 = B probably better · 0.5 = tie · 0.75 = A probably better · 1.0 = A clearly better")

    if consistent is not None:
        is_ok = str(consistent).lower() in ("true", "1")
        if is_ok:
            st.success("Both orderings agree on the winner.")
        else:
            st.warning("Orderings disagree or scores diverge significantly. Check reasoning below.")

    if comment_ab or comment_ba:
        with st.expander("Gemini reasoning", expanded=True):
            if comment_ab:
                st.markdown("**AB ordering** (Rollout A shown as Video 1)")
                st.info(comment_ab)
            if comment_ba:
                st.markdown("**BA ordering** (Rollout B shown as Video 1)")
                st.info(comment_ba)


# ── Browse tab ────────────────────────────────────────────────────────────────

def tab_browse(registry: pd.DataFrame) -> None:
    pairs = load_vlm_pairs()
    if pairs is None:
        st.info("No VLM pairs stored yet. Switch to **Live Gemini** to score your first pair.")
        return

    options = [f"{r['traj_A']}  vs  {r['traj_B']}" for _, r in pairs.iterrows()]
    idx = st.selectbox("Labeled pair", range(len(options)), format_func=lambda i: options[i])
    row = pairs.iloc[idx]

    st.caption(
        f"Scored {row.get('timestamp', '?')}  ·  "
        f"Model: `{row.get('model_version', '?')}`  ·  "
        f"Prompt: `{row.get('prompt_version', '?')}`"
    )
    st.divider()

    col_a, col_b = st.columns(2)
    render_video_col(col_a, "A", row["traj_A"], registry)
    render_video_col(col_b, "B", row["traj_B"], registry)

    st.divider()
    render_scores(
        label=float(row["label"]),
        score_ab=row.get("score_ab"),
        score_ba_flipped=row.get("score_ba_flipped"),
        comment_ab=row.get("comment_ab", ""),
        comment_ba=row.get("comment_ba", ""),
        consistent=row.get("consistent"),
    )

    sid = _sheets_id()
    if not sid and VLM_CSV.exists():
        st.divider()
        st.download_button("Download vlm_pairs.csv", data=VLM_CSV.read_bytes(), file_name="vlm_pairs.csv", mime="text/csv")


# ── Live tab ──────────────────────────────────────────────────────────────────

def tab_live(registry: pd.DataFrame, api_key: str, model: str) -> None:
    tasks = sorted(registry["task_name"].dropna().unique().tolist())
    selected_task = st.selectbox(
        "Filter by task",
        ["All tasks"] + tasks,
        format_func=lambda t: t.replace("_", " ").title() if t != "All tasks" else "All tasks",
        key="live_task_filter",
    )
    filtered = registry if selected_task == "All tasks" else registry[registry["task_name"] == selected_task]
    rollout_ids = filtered["rollout_id"].tolist()

    if len(rollout_ids) < 2:
        st.warning("Need at least 2 rollouts in the selected task to compare.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        rollout_a = st.selectbox("Rollout A", rollout_ids, key="live_a")
    with col_b:
        remaining = [r for r in rollout_ids if r != rollout_a]
        rollout_b = st.selectbox("Rollout B", remaining, key="live_b")

    vid_a, vid_b = st.columns(2)
    render_video_col(vid_a, "A", rollout_a, registry)
    render_video_col(vid_b, "B", rollout_b, registry)

    st.divider()

    if not api_key:
        st.warning("Go to **Settings** to add your Gemini API key before scoring.")

    if st.button("Run Gemini", type="primary", width="stretch", disabled=not api_key):
        row_a = registry[registry["rollout_id"] == rollout_a].iloc[0]
        row_b = registry[registry["rollout_id"] == rollout_b].iloc[0]

        def _resolve_video(row, label: str) -> str | None:
            """Return a local path to the video, downloading from HF if needed."""
            local = ROOT / row["video_path"]
            if local.exists():
                return str(local)
            url = row.get("video_url", "")
            if url:
                with st.spinner(f"Downloading Rollout {label} from HuggingFace..."):
                    r = requests.get(url, timeout=120)
                    r.raise_for_status()
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(r.content)
                        return tmp.name
            return None

        path_a = _resolve_video(row_a, "A")
        path_b = _resolve_video(row_b, "B")

        if not path_a:
            st.error("Video for Rollout A not found locally and no video_url in registry. Run patch_registry_video_urls.py first.")
            return
        if not path_b:
            st.error("Video for Rollout B not found locally and no video_url in registry. Run patch_registry_video_urls.py first.")
            return

        from src.preferences.vlm_gemini import PROMPT_VERSION, label_pair

        with st.spinner(f"Uploading videos and querying {model}... (~20-40s)"):
            try:
                result = label_pair(str(path_a), str(path_b), api_key=api_key, model=model)
            except Exception as exc:
                st.error(f"Gemini call failed: {exc}")
                return

        st.success("Gemini responded.")
        render_scores(
            label=result["score"],
            score_ab=result["score_ab"],
            score_ba_flipped=result["score_ba_flipped"],
            comment_ab=result["comment_ab"],
            comment_ba=result["comment_ba"],
            consistent=result["consistent"],
        )
        st.session_state["live_result"] = result
        st.session_state["live_pair"] = (rollout_a, rollout_b, model, PROMPT_VERSION)

    if "live_result" in st.session_state:
        result = st.session_state["live_result"]
        ra, rb, mdl, pv = st.session_state["live_pair"]
        st.write("")
        if st.button("Save to vlm_pairs", width="stretch"):
            save_pair({
                "traj_A": ra, "traj_B": rb,
                "label": result["score"],
                "evaluator_type": "vlm_gemini",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model_version": mdl,
                "prompt_version": pv,
                "score_ab": result["score_ab"],
                "score_ba_flipped": result["score_ba_flipped"],
                "comment_ab": result["comment_ab"],
                "comment_ba": result["comment_ba"],
                "consistent": result["consistent"],
            })
            st.success(f"Saved: `{ra}` vs `{rb}`  ->  label = {result['score']:.2f}")
            del st.session_state["live_result"]
            del st.session_state["live_pair"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        if st.button("Refresh rollouts", icon="🔄"):
            load_registry.clear()
            st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

def _resolve_api_key() -> str:
    if "gemini_api_key" not in st.session_state or not st.session_state["gemini_api_key"]:
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            try:
                key = st.secrets.get("GOOGLE_API_KEY", "")
            except Exception:
                pass
        st.session_state["gemini_api_key"] = key
    return st.session_state["gemini_api_key"]


def main() -> None:
    render_sidebar()

    api_key = _resolve_api_key()
    model = st.session_state.get("gemini_model", "gemini-3.6-flash")

    st.title("VLM Preference Viewer")
    st.caption("Explore stored Gemini labels or score a new pair live.")

    if not api_key:
        st.info("Add your Gemini API key in **Settings** or set `GOOGLE_API_KEY` in your `.env` file.", icon="⚙️")

    registry = load_registry()
    browse_tab, live_tab = st.tabs(["Browse Results", "Live Gemini"])

    with browse_tab:
        tab_browse(registry)
    with live_tab:
        tab_live(registry, api_key, model)


main()
