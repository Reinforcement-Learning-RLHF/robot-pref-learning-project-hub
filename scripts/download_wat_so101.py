"""
Download ammiellewb/wat.ai-so101 from HuggingFace, group frames by episode,
render MP4 videos, and register each episode as a rollout.

Label convention in the dataset:
    0  = failure
    1  = success
    -1 = unsure (skipped by default, pass --include-unsure to keep)

Usage:
    python scripts/download_wat_so101.py
    python scripts/download_wat_so101.py --fps 10 --include-unsure
    python scripts/download_wat_so101.py --limit 20   # first 20 episodes only
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROLLOUTS_DIR = ROOT / "data" / "rollouts"
REGISTRY_CSV = ROOT / "data" / "rollout_registry.csv"
DATASET_ID   = "ammiellewb/wat.ai-so101"
TASK_NAME    = "pour"
TASK_GOAL    = "Tilt the container to pour its liquid contents into the target cup."


# ── Helpers ───────────────────────────────────────────────────────────────────

def frames_to_mp4(frames: list, output_path: Path, fps: int) -> None:
    arr = np.stack([np.array(f.convert("RGB")) for f in frames]).astype(np.uint8)
    H, W = arr.shape[1], arr.shape[2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.read().decode())


def upsert_registry(meta: dict) -> None:
    fieldnames = [
        "rollout_id", "task_name", "task_goal", "video_filename", "video_path",
        "simulator", "success_label", "timestamp", "additional_notes",
    ]
    rows: list[dict] = []
    if REGISTRY_CSV.exists():
        with REGISTRY_CSV.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["rollout_id"] != meta["rollout_id"]]
    rows.append({k: meta.get(k, "") for k in fieldnames})
    with REGISTRY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=10,
                        help="Playback FPS (default 10 — frames are sparse)")
    parser.add_argument("--include-unsure", action="store_true",
                        help="Also include episodes labelled -1 (unsure)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N episodes")
    args = parser.parse_args()

    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"  {len(ds)} rows loaded.")

    # Parse label strings like "0 act_dataset_20260321_161742__episode_0"
    # Label names list is in ds.features["label"].names
    label_names = ds.features["label"].names  # e.g. ["0 act_dataset_..._episode_0", ...]

    # Group row indices by episode
    episodes: dict[str, dict] = {}
    for i, row in enumerate(ds):
        raw = label_names[row["label"]]        # e.g. "0 act_dataset_..._episode_3"
        parts = raw.split(" ", 1)
        numeric_label = int(parts[0])          # 0, 1, or -1
        episode_name  = parts[1] if len(parts) > 1 else raw

        if numeric_label == -1 and not args.include_unsure:
            continue

        if episode_name not in episodes:
            episodes[episode_name] = {"label": numeric_label, "frames": []}
        episodes[episode_name]["frames"].append(row["image"])

    episode_list = sorted(episodes.keys())
    if args.limit:
        episode_list = episode_list[:args.limit]

    print(f"\nEpisodes to convert: {len(episode_list)}")
    success_count = sum(1 for e in episode_list if episodes[e]["label"] == 1)
    failure_count = sum(1 for e in episode_list if episodes[e]["label"] == 0)
    unsure_count  = sum(1 for e in episode_list if episodes[e]["label"] == -1)
    print(f"  {success_count} success · {failure_count} failure · {unsure_count} unsure")

    created, skipped = 0, 0
    for ep_name in episode_list:
        ep = episodes[ep_name]
        numeric_label = ep["label"]
        frames = ep["frames"]

        # Sanitise rollout_id (replace spaces/special chars)
        rollout_id = ep_name.replace(" ", "_").replace("/", "_")
        rollout_dir = ROLLOUTS_DIR / rollout_id

        if (rollout_dir / "rollout_agentview.mp4").exists():
            print(f"  [skip] {rollout_id} already exists")
            skipped += 1
            continue

        rollout_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = rollout_dir / "rollout_agentview.mp4"

        print(f"  [{numeric_label:+d}] {rollout_id}  ({len(frames)} frames) ...", end=" ", flush=True)
        frames_to_mp4(frames, mp4_path, fps=args.fps)
        print("done")

        success_label = numeric_label == 1
        notes = {0: "Labelled failure (0)", 1: "Labelled success (1)", -1: "Labelled unsure (-1)"}.get(numeric_label, "")

        meta = {
            "rollout_id":   rollout_id,
            "task_name":    TASK_NAME,
            "task_goal":    TASK_GOAL,
            "video_filename": "rollout_agentview.mp4",
            "video_path":   f"data/rollouts/{rollout_id}/rollout_agentview.mp4",
            "simulator":    "real",
            "success_label": success_label,
            "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "additional_notes": notes,
        }
        (rollout_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        upsert_registry(meta)
        created += 1

    print(f"\nDone. {created} rollouts created, {skipped} skipped (already existed).")
    print(f"Registry: {REGISTRY_CSV} ({created + skipped} total rows added from this dataset)")


if __name__ == "__main__":
    main()
