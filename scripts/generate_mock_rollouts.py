"""
Generates mock rollout data to test the create_rollout_registry.py script.
Creates dummy directories, empty .mp4 files, and metadata.json files in data/rollouts/.
"""

import json
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    rollouts_dir = project_root / "data" / "rollouts"
    
    # Create the base directory if it doesn't exist
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    mocks = [
        {
            "folder": "mock_pour_001",
            "metadata": {
                "rollout_id": "mock_pour_001",
                "task_name": "pour",
                "task_goal": "Pour contents.",
                "video_filename": "rollout_agentview.mp4",
                "video_path": "data/rollouts/mock_pour_001/rollout_agentview.mp4",
                "simulator": "robosuite",
                "success_label": True,
                "timestamp": "2026-05-30T14:00:00Z",
                "additional_notes": "Perfect pour"
            }
        },
        {
            "folder": "mock_lift_002",
            "metadata": {
                "rollout_id": "mock_lift_002",
                "task_name": "lift",
                "task_goal": "Lift the block.",
                "video_filename": "rollout_wristview.mp4",
                "video_path": "data/rollouts/mock_lift_002/rollout_wristview.mp4",
                "simulator": "isaac",
                "success_label": False,
                "timestamp": "2026-05-30T15:00:00Z"
            } # 'additional_notes' intentionally omitted to test optional field handling
        },
        {
            "folder": "mock_missing_meta",
            "metadata": None # Will test the "[skip] ... no metadata.json found" branch
        },
        {
            "folder": "mock_invalid_json",
            "metadata": "{ this is not valid json, missing quotes and brackets }", # Tests JSON decoding error
        }
    ]

    for mock in mocks:
        mock_dir = rollouts_dir / mock["folder"]
        mock_dir.mkdir(exist_ok=True)

        # Create metadata.json
        if mock["metadata"] is not None:
            meta_path = mock_dir / "metadata.json"
            if isinstance(mock["metadata"], str):
                meta_path.write_text(mock["metadata"])
            else:
                with meta_path.open("w") as f:
                    json.dump(mock["metadata"], f, indent=2)

        # Create an empty dummy video file
        (mock_dir / "dummy_video.mp4").touch()

    print(f"✅ Created {len(mocks)} mock rollouts in {rollouts_dir.relative_to(project_root)}")

if __name__ == "__main__":
    main()