"""
Download lerobot/svla_so101_pickplace (v3 format) from HuggingFace and
convert each episode into our data/rollouts/ registry format.

The dataset stores all episodes in a single combined parquet + combined MP4
per chunk (lerobot v3). This script:
  - Reads meta/episodes/ for per-episode frame counts and video timestamps
  - Clips each episode out of the combined MP4 using ffmpeg
  - Writes per-episode metadata.json

Usage:
    python scripts/download_lerobot_dataset.py
    python scripts/download_lerobot_dataset.py --max-episodes 5
    python scripts/download_lerobot_dataset.py --max-episodes 5 --skip-video
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLLOUTS_DIR = PROJECT_ROOT / "data" / "rollouts"

DATASET_REPO = "lerobot/svla_so101_pickplace"

# Use the up-facing (top-down) camera as our agentview
CAMERA_KEY = "observation.images.up"


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _extract_clip(src: Path, dst: Path, t_start: float, t_end: float) -> bool:
    """
    Extract a time-range clip from src into dst using ffmpeg.
    Uses stream copy (no re-encode) for speed.
    Returns True on success.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(t_start),
        "-to", str(t_end),
        "-i", str(src),
        "-c", "copy",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def convert_episode(
    root: Path,
    ep: dict,
    skip_video: bool = False,
) -> bool:
    """
    Convert one episode row (from episodes metadata) to rollout format.
    Returns success_label (always True for curated demos).
    """
    episode_idx = int(ep["episode_index"])
    rollout_id = f"so101_episode_{episode_idx:06d}"
    rollout_dir = ROLLOUTS_DIR / rollout_id
    rollout_dir.mkdir(parents=True, exist_ok=True)

    steps = int(ep["length"])

    # --- video ---
    video_filename = ""
    video_path_rel = ""

    if not skip_video:
        chunk_idx = int(ep[f"videos/{CAMERA_KEY}/chunk_index"])
        file_idx   = int(ep[f"videos/{CAMERA_KEY}/file_index"])
        t_start    = float(ep[f"videos/{CAMERA_KEY}/from_timestamp"])
        t_end      = float(ep[f"videos/{CAMERA_KEY}/to_timestamp"])

        src_video = (
            root / "videos" / CAMERA_KEY
            / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        )
        if src_video.exists():
            video_filename  = f"{rollout_id}_agentview.mp4"
            video_path_rel  = f"data/rollouts/{rollout_id}/{video_filename}"
            dst_video = rollout_dir / video_filename
            ok = _extract_clip(src_video, dst_video, t_start, t_end)
            if not ok:
                print(f"    [warn] ffmpeg clip failed for episode {episode_idx}")
                video_filename = ""
                video_path_rel = ""
        else:
            print(f"    [warn] combined video not found: {src_video}")

    # --- metadata.json ---
    metadata: dict = {
        "rollout_id": rollout_id,
        "task_name": "pick_and_place",
        "task_goal": "Pick up the cube and place it in the target location.",
        "video_filename": video_filename,
        "video_path": video_path_rel,
        "simulator": "real_so101",
        "success_label": True,  # all episodes in this dataset are successful demonstrations
        "timestamp": "2026-03-01T00:00:00Z",
        "additional_notes": f"From {DATASET_REPO} dataset",
        "steps_recorded": steps,
        "source_dataset": DATASET_REPO,
        "episode_index": episode_idx,
    }

    (rollout_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return True  # curated demos are all successful


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert lerobot SO-101 pick-and-place dataset."
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Convert only the first N episodes (for testing). Omit for all 50.",
    )
    parser.add_argument(
        "--skip-video", action="store_true",
        help="Write metadata.json only, skip video extraction (useful for quick testing).",
    )
    args = parser.parse_args()

    # Check ffmpeg availability upfront
    if not args.skip_video and not _has_ffmpeg():
        print(
            "WARNING: ffmpeg not found on PATH. Video clips will not be extracted.\n"
            "Install ffmpeg or re-run with --skip-video to suppress this warning.\n"
        )
        args.skip_video = True

    print(f"Downloading {DATASET_REPO} from HuggingFace...")
    print("(files are cached in ~/.cache/huggingface/hub -- re-runs are instant)\n")
    local_dir = snapshot_download(repo_id=DATASET_REPO, repo_type="dataset")
    root = Path(local_dir)
    print(f"Dataset root: {root}\n")

    # Load per-episode metadata (length, video timestamps, etc.)
    ep_meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not ep_meta_path.exists():
        print(f"ERROR: episodes metadata not found at {ep_meta_path}")
        sys.exit(1)

    ep_df = pd.read_parquet(ep_meta_path).sort_values("episode_index").reset_index(drop=True)
    total_episodes = len(ep_df)
    print(f"Episodes in dataset: {total_episodes}")

    if args.max_episodes is not None:
        ep_df = ep_df.head(args.max_episodes)
        print(f"--max-episodes {args.max_episodes}: converting first {len(ep_df)} episode(s).")
    print()

    n_converted = 0
    for _, ep in ep_df.iterrows():
        episode_idx = int(ep["episode_index"])
        rollout_id = f"so101_episode_{episode_idx:06d}"
        print(f"  [{episode_idx + 1}/{total_episodes}] {rollout_id}  ({int(ep['length'])} steps)", end="  ")
        convert_episode(root, ep.to_dict(), skip_video=args.skip_video)
        n_converted += 1
        print("ok")

    print(f"\nDownloaded and converted {n_converted} episode(s)")
    print(f"  Successful: {n_converted}  (all episodes in this dataset are demonstrations)")
    print(f"  Failed:     0")
    if args.skip_video:
        print("  Videos: skipped (--skip-video)")
    print(f"\nSaved to {ROLLOUTS_DIR}")


if __name__ == "__main__":
    main()
