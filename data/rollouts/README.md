# Rollout Data

Place rollout subfolders here. Each subfolder must contain a `metadata.json` file.

See [docs/rollout_format.md](../../docs/rollout_format.md) for the full format specification.

## Quick Structure Reference

```
rollouts/
└── <rollout_id>/
    ├── metadata.json
    └── <video_filename>.mp4
```

Run `scripts/create_rollout_registry.py` from the project root to regenerate `data/rollout_registry.csv`.
