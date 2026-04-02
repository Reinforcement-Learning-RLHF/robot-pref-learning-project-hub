import h5py
import cv2
import numpy as np
import argparse

def hdf5_to_video(hdf5_path, dataset_key, output_path, fps=30):
    # Open HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset key '{dataset_key}' not found. Available keys: {list(f.keys())}")

        images = f[dataset_key][:]
        # Convert to uint8 if needed
        if images.dtype != np.uint8:
            images = (255 * images).astype(np.uint8)

        # Handle (T, C, H, W) -> (T, H, W, C)
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))

        T, H, W, C = images.shape
        print(f"Loaded {T} frames of size {H}x{W}x{C}")

        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")
    
    # Ensure uint8 format
    if images.dtype != np.uint8:
        images = (255 * images).astype(np.uint8)

    # Get dimensions
    T, H, W, C = images.shape
    print(f"Loaded {T} frames of size {H}x{W}x{C}")

    # OpenCV expects BGR
    if C == 3:
        convert_color = True
    else:
        raise ValueError("Only 3-channel (RGB) images supported")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(T):
        frame = images[i]

        if convert_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        writer.write(frame)

    writer.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", type=str, help="Path to HDF5 file")
    parser.add_argument("--key", type=str, default="observations/images/wrist_cam", help="Dataset key for images")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    hdf5_to_video(args.hdf5_path, args.key, args.output, args.fps)