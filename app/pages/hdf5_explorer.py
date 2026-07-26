"""
HDF5 Explorer — browse simulation episode files, preview frames, plot signals,
and convert episodes to the standard rollout format without leaving the app.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"


# ── File discovery ────────────────────────────────────────────────────────────

@st.cache_data
def find_hdf5_files() -> list[Path]:
    return sorted(DATA_DIR.rglob("*.hdf5")) + sorted(DATA_DIR.rglob("*.h5"))


# ── HDF5 helpers ──────────────────────────────────────────────────────────────

def _collect_items(f: h5py.File) -> list[dict]:
    items = []

    def _visit(name, obj):
        kind = "dataset" if isinstance(obj, h5py.Dataset) else "group"
        entry = {"path": name, "kind": kind}
        if kind == "dataset":
            entry["shape"] = obj.shape
            entry["dtype"] = str(obj.dtype)
        items.append(entry)

    f.visititems(_visit)
    return items


@st.cache_data
def load_dataset(file_path: str, dataset_path: str) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        return f[dataset_path][()]


def _is_image_dataset(shape: tuple, dtype: str) -> bool:
    """True for datasets shaped (T, C, H, W) or (T, H, W, C) with C in {1,3,4}."""
    if len(shape) != 4:
        return False
    return shape[1] in (1, 3, 4) or shape[3] in (1, 3, 4)


def _frames_to_hwc_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 4 and frames.shape[1] in (1, 3, 4):
        frames = frames.transpose(0, 2, 3, 1)
    if frames.dtype != np.uint8:
        frames = (frames * 255 if frames.max() <= 1.0 else frames).clip(0, 255).astype(np.uint8)
    if frames.shape[3] == 1:
        frames = np.repeat(frames, 3, axis=3)
    return frames


def _raw_to_mp4(raw_path: str, out_path: str, H: int, W: int, fps: int) -> None:
    """Encode a raw RGB24 file to H.264 MP4 via ffmpeg (no pipe, avoids Windows deadlock)."""
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", raw_path,
        "-vf", "scale=640:480:flags=lanczos",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "fast",
        out_path,
    ], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode()[-500:])


def _frames_to_mp4_bytes(file_path: str, dataset_path: str, fps: int = 30) -> bytes:
    """Read frames from HDF5, write raw file, encode with ffmpeg, return MP4 bytes."""
    with h5py.File(file_path, "r") as f:
        frames = _frames_to_hwc_uint8(f[dataset_path][()])
    _, H, W, _ = frames.shape

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as raw_tmp:
        raw_tmp.write(frames.tobytes())
        raw_path = raw_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as mp4_tmp:
        mp4_path = mp4_tmp.name

    try:
        _raw_to_mp4(raw_path, mp4_path, H, W, fps)
        return Path(mp4_path).read_bytes()
    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(mp4_path).unlink(missing_ok=True)


def _to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a single frame from CHW or HWC float/uint8 to HWC uint8."""
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = frame.transpose(1, 2, 0)  # CHW → HWC
    if frame.dtype != np.uint8:
        lo, hi = frame.min(), frame.max()
        if hi <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    return frame


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(files: list[Path]) -> Path | None:
    with st.sidebar:
        st.markdown("## HDF5 Files")
        if not files:
            st.warning("No .hdf5 / .h5 files in `data/`.")
            return None
        labels = [str(p.relative_to(ROOT)) for p in files]
        idx = st.selectbox("File", range(len(labels)), format_func=lambda i: labels[i])
        return files[idx]


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("🗂️ HDF5 Explorer")

files = find_hdf5_files()
selected_file = render_sidebar(files)

if not files:
    st.info("Put `.hdf5` / `.h5` files anywhere under `data/` and refresh.", icon="📂")
    st.stop()

if selected_file is None:
    st.stop()

st.caption(f"`{selected_file.relative_to(ROOT)}`  ·  {selected_file.stat().st_size / 1024:.1f} KB")

with h5py.File(str(selected_file), "r") as f:
    items = _collect_items(f)
    root_attrs = dict(f.attrs)

datasets = [it for it in items if it["kind"] == "dataset"]

if root_attrs:
    with st.expander("File attributes"):
        st.json({k: str(v) for k, v in root_attrs.items()})

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_struct, tab_frames, tab_signals, tab_batch = st.tabs(["Structure", "Camera frames", "Signals", "Batch convert"])

# ── Structure tab ─────────────────────────────────────────────────────────────

with tab_struct:
    rows = []
    for it in items:
        indent = "  " * it["path"].count("/")
        icon = "📊" if it["kind"] == "dataset" else "📁"
        rows.append({
            "Name": f"{indent}{icon} {it['path'].split('/')[-1]}",
            "Full path": it["path"],
            "Shape": str(it.get("shape", "")) if it["kind"] == "dataset" else "",
            "dtype": it.get("dtype", "") if it["kind"] == "dataset" else "",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ── Camera frames tab ─────────────────────────────────────────────────────────

with tab_frames:
    image_datasets = [
        it for it in datasets
        if _is_image_dataset(it["shape"], it["dtype"])
    ]

    if not image_datasets:
        st.info("No image datasets detected (expected shape T×C×H×W or T×H×W×C).")
    else:
        cam_paths = [it["path"] for it in image_datasets]
        cam_labels = [p.split("/")[-1] for p in cam_paths]

        n_timesteps = image_datasets[0]["shape"][0]
        t = st.slider("Timestep", 0, n_timesteps - 1, 0, key="frame_slider")

        cols = st.columns(len(cam_paths))
        for col, path, label in zip(cols, cam_paths, cam_labels):
            arr = load_dataset(str(selected_file), path)
            frame = _to_hwc_uint8(arr[t])
            with col:
                st.image(frame, caption=label, width="stretch")

        st.caption(f"Episode length: {n_timesteps} timesteps")

        st.divider()
        st.subheader("Preview as video")
        prev_fps = st.slider("FPS", 5, 60, 30, key="preview_fps")
        prev_cam = st.selectbox("Camera", cam_paths, format_func=lambda p: p.split("/")[-1], key="preview_cam")

        if st.button("Generate video preview", key="preview_btn"):
            with st.spinner("Rendering MP4..."):
                try:
                    video_bytes = _frames_to_mp4_bytes(str(selected_file), prev_cam, fps=prev_fps)
                    st.session_state["preview_video"] = video_bytes
                    st.session_state["preview_cam_label"] = prev_cam.split("/")[-1]
                except Exception as exc:
                    st.error(f"Failed: {exc}")

        if "preview_video" in st.session_state:
            st.video(st.session_state["preview_video"], format="video/mp4")
            st.download_button(
                f"Download {st.session_state['preview_cam_label']}.mp4",
                data=st.session_state["preview_video"],
                file_name=f"{selected_file.stem}_{st.session_state['preview_cam_label']}.mp4",
                mime="video/mp4",
            )

        st.divider()
        st.subheader("Convert to rollout")

        with st.form("convert_form"):
            col1, col2 = st.columns(2)
            with col1:
                task = st.text_input("Task name", value="pick_place",
                                     help="e.g. pick_place, pour, lift")
                task_goal = st.text_input("Task goal (optional)",
                                          placeholder="Pick the cube and place it in the bin.")
            with col2:
                success = st.selectbox("Success label", ["true", "false"])
                rollout_id = st.text_input("Rollout ID (optional)",
                                           placeholder=f"{selected_file.stem}",
                                           help="Leave blank to auto-generate from task + filename")
            notes = st.text_input("Notes (optional)", placeholder="e.g. clean sim run, no contact failures")
            fps = st.number_input("FPS", min_value=1, max_value=120, value=30)
            submitted = st.form_submit_button("Convert to MP4 rollout", type="primary", width="stretch")

        if submitted:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "hdf5_to_rollout.py"),
                str(selected_file),
                "--task", task,
                "--success", success,
                "--fps", str(fps),
            ]
            if rollout_id.strip():
                cmd += ["--rollout-id", rollout_id.strip()]
            if task_goal.strip():
                cmd += ["--task-goal", task_goal.strip()]
            if notes.strip():
                cmd += ["--notes", notes.strip()]

            with st.spinner("Converting... this may take 10-30 seconds"):
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

            if result.returncode == 0:
                st.success("Rollout created successfully!", icon="✅")
                st.code(result.stdout, language="text")
                find_hdf5_files.clear()
            else:
                st.error("Conversion failed.")
                st.code(result.stderr, language="text")

# ── Signals tab ──────────────────────────────────────────────────────────────

with tab_signals:
    numeric_datasets = [
        it for it in datasets
        if it["dtype"].startswith("float") or it["dtype"].startswith("int")
        if not _is_image_dataset(it["shape"], it["dtype"])
        if len(it["shape"]) in (1, 2)
    ]

    if not numeric_datasets:
        st.info("No 1-D or 2-D numeric (non-image) datasets found.")
    else:
        ds_options = [it["path"] for it in numeric_datasets]
        chosen = st.selectbox("Dataset", ds_options)
        data = load_dataset(str(selected_file), chosen)

        c1, c2 = st.columns(2)
        c1.json({"shape": list(data.shape), "dtype": str(data.dtype)})
        if data.size:
            flat = data.flatten().astype(float)
            c2.json({
                "min": round(float(np.min(flat)), 5),
                "max": round(float(np.max(flat)), 5),
                "mean": round(float(np.mean(flat)), 5),
                "std": round(float(np.std(flat)), 5),
            })

        st.divider()

        if data.ndim == 2 and data.shape[0] > 1:
            n_dims = data.shape[1]
            dim_labels = [f"dim_{i}" for i in range(n_dims)]
            selected_dims = st.multiselect(
                "Dimensions",
                dim_labels,
                default=dim_labels[:min(6, n_dims)],
            )
            if selected_dims:
                indices = [int(d.split("_")[1]) for d in selected_dims]
                st.line_chart(
                    pd.DataFrame(data[:, indices], columns=selected_dims),
                    width="stretch",
                )
        elif data.ndim == 1 and data.shape[0] > 1:
            st.line_chart(pd.DataFrame({"value": data}), width="stretch")

        n_rows = st.slider("Preview rows", 5, min(100, len(data)), 10)
        if data.ndim == 1:
            st.dataframe(pd.DataFrame({"value": data[:n_rows]}), width="stretch", hide_index=True)
        else:
            st.dataframe(
                pd.DataFrame(data[:n_rows], columns=[f"dim_{i}" for i in range(data.shape[1])]),
                width="stretch", hide_index=True,
            )

# ── Batch convert tab ─────────────────────────────────────────────────────────

with tab_batch:
    st.caption("Convert multiple HDF5 episodes to rollouts in one go.")

    all_files = find_hdf5_files()
    if not all_files:
        st.info("No HDF5 files found under `data/`.")
    else:
        # Shared settings
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            batch_task = st.text_input("Task name (all episodes)", value="pick_place", key="batch_task")
        with bc2:
            batch_goal = st.text_input("Task goal (optional)", key="batch_goal",
                                        placeholder="Pick the cube and place it in the bin.")
        with bc3:
            batch_fps = st.number_input("FPS", min_value=1, max_value=120, value=30, key="batch_fps")

        st.divider()
        st.markdown("**Label each episode:**")

        # Per-episode success label
        episode_labels: dict[str, str] = {}
        skip_flags: dict[str, bool] = {}

        header_cols = st.columns([3, 2, 1])
        header_cols[0].markdown("**File**")
        header_cols[1].markdown("**Success**")
        header_cols[2].markdown("**Skip**")

        for f in all_files:
            row_cols = st.columns([3, 2, 1])
            row_cols[0].caption(str(f.relative_to(ROOT)))
            safe_key = str(f.relative_to(ROOT)).replace("\\", "_").replace("/", "_").replace(".", "_")
            episode_labels[str(f)] = row_cols[1].selectbox(
                "Success", ["true", "false"], key=f"label_{safe_key}", label_visibility="collapsed"
            )
            skip_flags[str(f)] = row_cols[2].checkbox(
                "Skip", key=f"skip_{safe_key}", label_visibility="collapsed"
            )

        st.divider()
        to_convert = [f for f in all_files if not skip_flags[str(f)]]
        st.caption(f"{len(to_convert)} of {len(all_files)} episodes selected for conversion.")

        if st.button("Convert all selected", type="primary", width="stretch", key="batch_run"):
            import csv as _csv
            import json as _json
            from datetime import datetime, timezone

            ROLLOUTS_DIR = ROOT / "data" / "rollouts"
            REGISTRY_CSV = ROOT / "data" / "rollout_registry.csv"
            REGISTRY_FIELDS = [
                "rollout_id", "task_name", "task_goal", "video_filename",
                "video_path", "simulator", "success_label", "timestamp", "additional_notes", "video_url",
            ]

            def _upsert_registry(meta: dict) -> None:
                rows: list[dict] = []
                if REGISTRY_CSV.exists():
                    with REGISTRY_CSV.open(encoding="utf-8") as f:
                        rows = [r for r in _csv.DictReader(f) if r["rollout_id"] != meta["rollout_id"]]
                rows.append({k: meta.get(k, "") for k in REGISTRY_FIELDS})
                with REGISTRY_CSV.open("w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
                    w.writeheader(); w.writerows(rows)

            def _convert_one(hdf5_path: Path, task: str, goal: str, success: bool, fps: int) -> str:
                stem = hdf5_path.stem
                parent_tag = hdf5_path.parent.name if hdf5_path.parent.name != "hdf5_datasets" else ""
                rollout_id = f"{task}_{'_'.join(filter(None,[parent_tag,stem]))}"
                rollout_dir = ROLLOUTS_DIR / rollout_id
                mp4_path = rollout_dir / "rollout_agentview.mp4"

                if mp4_path.exists():
                    return f"⏭ {hdf5_path.name} (already exists)"

                rollout_dir.mkdir(parents=True, exist_ok=True)

                with h5py.File(str(hdf5_path), "r") as f:
                    frames = _frames_to_hwc_uint8(f["observations/images/main_observation"][()])
                _, H, W, _ = frames.shape

                with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as raw_tmp:
                    raw_tmp.write(frames.tobytes())
                    raw_path = raw_tmp.name
                del frames

                try:
                    _raw_to_mp4(raw_path, str(mp4_path), H, W, fps)
                finally:
                    Path(raw_path).unlink(missing_ok=True)

                meta = {
                    "rollout_id": rollout_id, "task_name": task, "task_goal": goal,
                    "video_filename": "rollout_agentview.mp4",
                    "video_path": f"data/rollouts/{rollout_id}/rollout_agentview.mp4",
                    "simulator": "sim", "success_label": success,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "additional_notes": "",
                }
                (rollout_dir / "metadata.json").write_text(_json.dumps(meta, indent=2))
                _upsert_registry(meta)
                return f"✅ {hdf5_path.name} → {rollout_id}"

            progress = st.progress(0, text="Starting...")
            log_area = st.empty()
            logs: list[str] = []
            errors: list[str] = []
            goal = batch_goal.strip() or f"Complete the {batch_task.replace('_', ' ')} task."

            for i, f in enumerate(to_convert):
                progress.progress(i / len(to_convert), text=f"{f.name} ({i+1}/{len(to_convert)})")
                try:
                    msg = _convert_one(f, batch_task, goal, episode_labels[str(f)] == "true", int(batch_fps))
                    logs.append(msg)
                except Exception as exc:
                    logs.append(f"❌ {f.name}")
                    errors.append(f"{f.name}: {exc}")
                log_area.code("\n".join(logs), language="text")

            progress.progress(1.0, text="Done!")
            find_hdf5_files.clear()

            if errors:
                st.error(f"{len(errors)} failed:")
                for e in errors:
                    st.code(e, language="text")
            else:
                st.success(f"All {len(to_convert)} episodes converted!", icon="🎉")
