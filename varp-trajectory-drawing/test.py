import h5py

def get_episode_timesteps(file_path, episode_group_path):
    """
    Returns the number of timesteps in a specific HDF5 episode.
    
    file_path: str, path to the .hdf5 file
    episode_group_path: str, internal path to the episode group (e.g., 'episode_0')
    """
    with h5py.File(file_path, 'r') as f:
        # Navigate to the specific episode group
        episode = f[episode_group_path]
        
        # Check for a standard dataset like 'actions' or 'obs' to get the length
        if 'actions' in episode:
            return len(episode['actions'])
        elif 'observations' in episode:
            return len(episode['observations'])
        else:
            print("what")

# Example Usage:
file = "collected_trajectories/"
path = "collected_trajectories/ep_0001_1773963178_6354215"
print(f"Timesteps: {get_episode_timesteps(file, path)}")

