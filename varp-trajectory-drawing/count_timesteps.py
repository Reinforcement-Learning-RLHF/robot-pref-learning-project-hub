import h5py
from pathlib import Path

trajectories_dir = Path(__file__).parent / "collected_trajectories"

total = 0
for ep_dir in sorted(trajectories_dir.iterdir()):
    hdf5_path = ep_dir / "demo.hdf5"
    if not hdf5_path.exists():
        continue
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        ep_timesteps = sum(f["data"][d]["actions"].shape[0] for d in demos)
    print(f"{ep_dir.name}: {ep_timesteps} timesteps")
    total += ep_timesteps

print(f"\nTotal: {total} timesteps across {len(list(trajectories_dir.iterdir()))} episodes")
