"""
Human preference labeling.
"""

import csv
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REGISTRY_CSV = ROOT / "data" / "rollout_registry.csv"
OUTPUT_CSV   = ROOT / "data" / "preferences" / "human_pairs.csv"
HUMAN_SHEET  = "human_pairs"

FIELDNAMES = [
    "traj_A", "traj_B", "label",
    "evaluator_type", "timestamp", "model_version", "prompt_version",
]


# ── Storage helpers ───────────────────────────────────────────────────────────

def _sheets_id():
    from src.storage.sheets import spreadsheet_id
    return spreadsheet_id()


def _hf_token():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            token = st.secrets["huggingface"].get("hf_token", "")
        except Exception:
            pass
    return token or None


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def load_registry() -> pd.DataFrame:
    df = pd.read_csv(REGISTRY_CSV, encoding="cp1252")
    df["success_label"] = df["success_label"].astype(str).str.lower() == "true"
    if "video_url" not in df.columns:
        df["video_url"] = ""
    return df


@st.cache_data
def load_video_bytes(path_str: str, video_url: str | None = None) -> bytes | None:
    path = Path(path_str)
    if path.exists():
        src_path, is_remote = str(path), False
    elif video_url:
        headers = {"Authorization": f"Bearer {_hf_token()}"} if _hf_token() else {}
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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out:
        out_path = out.name

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
    return path.read_bytes() if not is_remote and path.exists() else None


@st.cache_data(ttl=30)
def load_labeled_set() -> set[tuple[str, str]]:
    sid = _sheets_id()
    if sid:
        from src.storage.sheets import read_rows
        rows = read_rows(sid, HUMAN_SHEET)
    else:
        if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
            return set()
        with OUTPUT_CSV.open() as f:
            rows = list(csv.DictReader(f))
    labeled: set[tuple[str, str]] = set()
    for row in rows:
        a, b = row["traj_A"], row["traj_B"]
        labeled.add((a, b))
        labeled.add((b, a))
    return labeled


def compute_pairs(df: pd.DataFrame, labeled: set) -> list[tuple[dict, dict]]:
    successes = df[df["success_label"]].to_dict("records")
    failures  = df[~df["success_label"]].to_dict("records")
    return [
        (s, f) for s in successes for f in failures
        if (s["rollout_id"], f["rollout_id"]) not in labeled
    ]


def save_label(traj_a: str, traj_b: str, label: float) -> None:
    row = {
        "traj_A": traj_a, "traj_B": traj_b, "label": label,
        "evaluator_type": "human",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": "human", "prompt_version": "n/a",
    }
    sid = _sheets_id()
    if sid:
        from src.storage.sheets import append_row
        append_row(sid, HUMAN_SHEET, row, FIELDNAMES)
    else:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
        with OUTPUT_CSV.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    load_labeled_set.clear()


# ── UI components ─────────────────────────────────────────────────────────────

def render_video(col, label: str, row: dict) -> None:
    with col:
        with st.container(border=True):
            st.markdown(f"#### Rollout {label}")
            st.caption(f"`{row['rollout_id']}`")
            video_path = ROOT / row["video_path"]
            data = load_video_bytes(str(video_path), row.get("video_url") or None)
            if data:
                st.video(data, format="video/mp4")
            else:
                st.warning(f"Video not found: `{row['video_path']}`")
            with st.expander("Notes"):
                st.write(row.get("additional_notes") or "n/a")


def render_sidebar(registry: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("## Dataset")
        total = len(registry)
        n_success = int(registry["success_label"].sum())
        n_fail = total - n_success
        c1, c2 = st.columns(2)
        c1.metric("Rollouts", total)
        c2.metric("Tasks", registry["task_name"].nunique())
        st.caption(f"{n_success} successes · {n_fail} failures")


# ── Page ──────────────────────────────────────────────────────────────────────

registry = load_registry()
render_sidebar(registry)

st.title("🏷️ Human Preference Labeling")

tasks = sorted(registry["task_name"].dropna().unique().tolist())
selected_task = st.selectbox(
    "Task",
    ["All tasks"] + tasks,
    format_func=lambda t: t.replace("_", " ").title() if t != "All tasks" else "All tasks",
)
df = registry if selected_task == "All tasks" else registry[registry["task_name"] == selected_task]

labeled = load_labeled_set()
pairs   = compute_pairs(df, labeled)

n_success = int(df["success_label"].sum())
n_failure = int((~df["success_label"]).sum())
total_possible = n_success * n_failure
labeled_count  = total_possible - len(pairs)

progress = labeled_count / total_possible if total_possible > 0 else 0.0
st.progress(progress, text=f"{labeled_count} / {total_possible} pairs labeled")
st.write("")

if total_possible == 0:
    st.warning("No success/failure pairs for this task.")
    st.stop()

if not pairs:
    st.success("All pairs labeled for this task! 🎉")
    st.stop()

if "row_a" not in st.session_state:
    chosen = random.choice(pairs)
    st.session_state.row_a = chosen[0]
    st.session_state.row_b = chosen[1]

row_a: dict = st.session_state.row_a
row_b: dict = st.session_state.row_b

task_display = row_a["task_name"].replace("_", " ").upper()
st.markdown(f'<div class="task-badge">{task_display}</div>', unsafe_allow_html=True)
st.caption(f"Goal: {row_a['task_goal']}")
st.divider()

col_a, col_b = st.columns(2)
render_video(col_a, "A", row_a)
render_video(col_b, "B", row_b)

st.divider()
st.markdown("**Which rollout better achieved the task goal?**")
st.write("")

btn_a, btn_tie, btn_b, _, skip_col = st.columns([2, 2, 2, 1, 1])


def on_label(label: float) -> None:
    save_label(row_a["rollout_id"], row_b["rollout_id"], label)
    del st.session_state.row_a
    del st.session_state.row_b


with btn_a:
    if st.button("👈  A is better", width="stretch", type="primary"):
        on_label(1.0); st.rerun()

with btn_tie:
    if st.button("🤝  Tie", width="stretch"):
        on_label(0.5); st.rerun()

with btn_b:
    if st.button("B is better  👉", width="stretch", type="primary"):
        on_label(0.0); st.rerun()

with skip_col:
    if st.button("Skip →", width="stretch", help="Skip without labeling"):
        del st.session_state.row_a
        del st.session_state.row_b
        st.rerun()
