"""
Scans data/rollouts/ for rollout subfolders, reads each metadata.json,
and writes/updates data/rollout_registry.csv.

Usage:
    python scripts/create_rollout_registry.py
    python scripts/create_rollout_registry.py --rollouts-dir path/to/rollouts
    python scripts/create_rollout_registry.py --output data/rollout_registry.csv
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_COLUMNS = [
    "rollout_id",
    "task_name",
    "task_goal",
    "video_filename",
    "video_path",
    "simulator",
    "success_label",
    "timestamp",
    "additional_notes",
]

REQUIRED_FIELDS = {
    "rollout_id",
    "task_name",
    "task_goal",
    "video_filename",
    "video_path",
    "simulator",
    "success_label",
    "timestamp",
}


def load_metadata(rollout_dir: Path) -> dict | None:
    metadata_path = rollout_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  [skip] {rollout_dir.name}: no metadata.json found")
        return None

    try:
        with metadata_path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [skip] {rollout_dir.name}: invalid JSON — {e}")
        return None

    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        print(f"  [warn] {rollout_dir.name}: missing required fields: {sorted(missing)}")

    row = {col: "" for col in REGISTRY_COLUMNS}
    for col in REGISTRY_COLUMNS:
        if col in data:
            value = data[col]
            # Normalise success_label to a consistent string
            if col == "success_label":
                row[col] = str(value).lower() if isinstance(value, bool) else str(value)
            else:
                row[col] = str(value)

    # Fall back: if rollout_id is missing, use the folder name
    if not row["rollout_id"]:
        row["rollout_id"] = rollout_dir.name
        print(f"  [warn] {rollout_dir.name}: rollout_id not in metadata.json, using folder name")

    return row


def scan_rollouts(rollouts_dir: Path) -> list[dict]:
    if not rollouts_dir.exists():
        print(f"Error: rollouts directory not found: {rollouts_dir}")
        sys.exit(1)

    rows = []
    subdirs = sorted(p for p in rollouts_dir.iterdir() if p.is_dir())

    if not subdirs:
        print(f"No rollout subfolders found in {rollouts_dir}")
        return rows

    for rollout_dir in subdirs:
        row = load_metadata(rollout_dir)
        if row is not None:
            rows.append(row)

    return rows


def write_registry(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Build rollout_registry.csv from rollout folders.")
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=project_root / "data" / "rollouts",
        help="Path to the rollouts folder (default: data/rollouts/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "rollout_registry.csv",
        help="Output CSV path (default: data/rollout_registry.csv)",
    )
    args = parser.parse_args()

    print(f"Scanning: {args.rollouts_dir}")
    rows = scan_rollouts(args.rollouts_dir)

    if not rows:
        print("No valid rollouts found. Registry not written.")
        return

    write_registry(rows, args.output)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Wrote {len(rows)} rollout(s) to {args.output} at {generated_at}")


if __name__ == "__main__":
    main()
