"""
Upload individual episode MP4 clips to HuggingFace and update rollout_registry.csv with video_url.

Usage:
    python scripts/upload_clips_to_hf.py [--repo REPO_ID] [--dry-run]

Defaults to repo: nafisatibrahim/wat.ai-so101-videos
Uploads each data/rollouts/<rollout_id>/rollout_agentview.mp4 as:
    episodes/<rollout_id>.mp4

After upload, updates video_url in data/rollout_registry.csv.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "rollout_registry.csv"

DEFAULT_REPO = "nafisatibrahim/wat.ai-so101-videos"


def hf_token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def resolve_url(repo_id: str, path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace dataset repo ID")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    parser.add_argument("--task", default=None, help="Only upload rollouts with this task_name")
    args = parser.parse_args()

    token = hf_token()
    if not token and not args.dry_run:
        print("ERROR: No HF token found. Run `huggingface-cli login` or set HF_TOKEN env var.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        print(f"Uploading to: https://huggingface.co/datasets/{args.repo}")

    with REGISTRY.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    skipped = 0
    for row in rows:
        if args.task and row["task_name"] != args.task:
            continue

        local = ROOT / row["video_path"] if row["video_path"] else None
        if not local or not local.exists():
            skipped += 1
            continue

        # Skip if already has a video_url that isn't empty/nan
        existing_url = row.get("video_url", "")
        if existing_url and str(existing_url) not in ("", "nan"):
            print(f"  SKIP (has url): {row['rollout_id']}")
            skipped += 1
            continue

        path_in_repo = f"episodes/{row['rollout_id']}.mp4"
        target_url = resolve_url(args.repo, path_in_repo)

        if args.dry_run:
            print(f"  [DRY RUN] {row['rollout_id']} -> {path_in_repo}")
        else:
            print(f"  Uploading {row['rollout_id']} ({local.stat().st_size // 1024}KB)...", end=" ", flush=True)
            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=path_in_repo,
                    repo_id=args.repo,
                    repo_type="dataset",
                )
                print("OK")
            except Exception as exc:
                print(f"FAILED: {exc}")
                continue

        row["video_url"] = target_url
        updated += 1

    print(f"\nUpdated {updated} video_url fields. Skipped {skipped}.")

    if not args.dry_run and updated > 0:
        with REGISTRY.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Registry saved: {REGISTRY}")
        print(f"\nDon't forget to add hf_token to .streamlit/secrets.toml:")
        print("[huggingface]")
        print("hf_token = \"hf_...\"")


if __name__ == "__main__":
    main()
