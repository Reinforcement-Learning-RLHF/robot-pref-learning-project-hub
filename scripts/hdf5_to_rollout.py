"""
Convert a simulation HDF5 episode file into the standard rollout format:

    data/rollouts/<rollout_id>/
        rollout_agentview.mp4
        rollout_wristview.mp4   (if wrist_cam exists)
        metadata.json

Then re-generates rollout_registry.csv.

Usage:
    python scripts/hdf5_to_rollout.py data/hdf5_datasets/episode_20.hdf5 \
        --task pick_place \
        --success true \
        --notes "Sim rollout from episode_20"

Required args:
    hdf5_path   Path to the .hdf5 file
    --task      task_name  (e.g. pick_place, pour, lift)
    --success   true / false

Optional:
    --rollout-id   Override auto-generated rollout ID
    --fps          Output video frame rate (default: 30)
    --notes        Free-text additional_notes field
    --camera       HDF5 path for main camera  (default: observations/images/main_observation)
    --wrist        HDF5 path for wrist camera (default: observations/images/wrist_cam)
    --no-wrist     Skip wrist cam even if present
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROLLOUTS_DIR = ROOT / "data" / "rollouts"
REGISTRY_CSV = ROOT / "data" / "rollout_registry.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = frame.transpose(1, 2, 0)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    return frame


def frames_to_mp4(frames: np.ndarray, output_path: Path, fps: int = 30) -> None:
    """Write (T, C, H, W) or (T, H, W, C) float/uint8 frames to H.264 MP4."""
    if frames.ndim == 4 and frames.shape[1] in (1, 3, 4):
        frames = frames.transpose(0, 2, 3, 1)
    if frames.dtype != np.uint8:
        frames = (frames * 255 if frames.max() <= 1.0 else frames).clip(0, 255).astype(np.uint8)
    if frames.shape[3] == 1:
        frames = np.repeat(frames, 3, axis=3)
    _, H, W, _ = frames.shape

    import tempfile as _tf
    with _tf.NamedTemporaryFile(suffix=".raw", delete=False) as raw_tmp:
        raw_tmp.write(frames.tobytes())
        raw_path = raw_tmp.name

    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
            "-r", str(fps), "-i", raw_path,
            "-vf", "scale=640:480:flags=lanczos",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            str(output_path),
        ], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")
    finally:
        Path(raw_path).unlink(missing_ok=True)


def update_registry(rollout_dir: Path, meta: dict) -> None:
    import csv

    fieldnames = [
        "rollout_id", "task_name", "task_goal", "video_filename", "video_path",
        "simulator", "success_label", "timestamp", "additional_notes",
    ]

    rows: list[dict] = []
    if REGISTRY_CSV.exists():
        with REGISTRY_CSV.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["rollout_id"] != meta["rollout_id"]]

    rows.append({k: meta.get(k, "") for k in fieldnames})

    with REGISTRY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Registry updated ({len(rows)} total rows) -> {REGISTRY_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HDF5 episode to rollout format.")
    parser.add_argument("hdf5_path", type=Path)
    parser.add_argument("--task", required=True, help="task_name, e.g. pick_place")
    parser.add_argument("--success", required=True, choices=["true", "false"])
    parser.add_argument("--rollout-id", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--notes", default="")
    parser.add_argument("--task-goal", default="")
    parser.add_argument("--camera", default="observations/images/main_observation")
    parser.add_argument("--wrist", default="observations/images/wrist_cam")
    parser.add_argument("--no-wrist", action="store_true")
    args = parser.parse_args()

    hdf5_path = ROOT / args.hdf5_path if not args.hdf5_path.is_absolute() else args.hdf5_path
    if not hdf5_path.exists():
        sys.exit(f"File not found: {hdf5_path}")

    stem = hdf5_path.stem  # e.g. "episode_20"
    rollout_id = args.rollout_id or f"{args.task}_{stem}"
    rollout_dir = ROLLOUTS_DIR / rollout_id
    rollout_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {hdf5_path.name} ...")
    with h5py.File(str(hdf5_path), "r") as f:
        # Main camera
        if args.camera not in f:
            available = []
            f.visititems(lambda n, o: available.append(n) if isinstance(o, h5py.Dataset) else None)
            sys.exit(
                f"Camera dataset '{args.camera}' not found.\n"
                f"Available datasets: {available}"
            )
        main_frames = f[args.camera][()]
        print(f"  main camera  {main_frames.shape}  {main_frames.dtype}")

        # Wrist camera (optional)
        wrist_frames = None
        if not args.no_wrist and args.wrist in f:
            wrist_frames = f[args.wrist][()]
            print(f"  wrist camera {wrist_frames.shape}  {wrist_frames.dtype}")

    # Write main video
    main_mp4 = rollout_dir / "rollout_agentview.mp4"
    print(f"Writing {main_mp4.name} ({len(main_frames)} frames @ {args.fps} fps) ...")
    frames_to_mp4(main_frames, main_mp4, fps=args.fps)
    print(f"  -> {main_mp4}")

    video_filename = main_mp4.name
    video_path = str(main_mp4.relative_to(ROOT)).replace("\\", "/")

    # Write wrist video
    if wrist_frames is not None:
        wrist_mp4 = rollout_dir / "rollout_wristview.mp4"
        print(f"Writing {wrist_mp4.name} ...")
        frames_to_mp4(wrist_frames, wrist_mp4, fps=args.fps)
        print(f"  -> {wrist_mp4}")

    # metadata.json
    meta = {
        "rollout_id": rollout_id,
        "task_name": args.task,
        "task_goal": args.task_goal or f"Complete the {args.task.replace('_', ' ')} task.",
        "video_filename": video_filename,
        "video_path": video_path,
        "simulator": "sim",
        "success_label": args.success == "true",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "additional_notes": args.notes,
    }
    meta_path = rollout_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  metadata -> {meta_path}")

    update_registry(rollout_dir, meta)

    print(f"\nDone. Rollout '{rollout_id}' is ready at:\n  {rollout_dir}")
    print("\nNext: run the registry patch script to add video_url if deploying to Streamlit Cloud:")
    print("  python scripts/patch_registry_video_urls.py")


if __name__ == "__main__":
    main()
