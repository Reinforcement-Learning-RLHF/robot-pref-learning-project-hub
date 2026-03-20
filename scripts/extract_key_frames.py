import cv2
import numpy as np
import ruptures as rpt
import os
from tqdm import tqdm
import argparse

# ----------------------------
# Parameters
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Key frame extraction")

    parser.add_argument("--video_path", type=str, default="demos/good_clean_slow_agentview.mp4",
                        help="where your video at")
    parser.add_argument("--output_dir", type=str, default="keyframes",
                        help="where to save key frames")
    parser.add_argument("--smooth_kernel", type=int, default=3,
                        help="smoothing kernel size for motion scoring")
    parser.add_argument("--penalty", type=float, default=2,
                        help="change-point detection sensitivity")
    parser.add_argument("--angle_weight", type=float, default=50,
                        help="rotation sensitivity")
    parser.add_argument("--mag_threshold", type=float, default=1,
                        help="threshold between moving and unmoving parts, helps mask out background")
    parser.add_argument("--min_seg_len", type=int, default=5,
                        help="min frames per segment")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    prev_gray = None
    motion_scores = []
    frames = []

    pbar = tqdm(total=0, unit=" frame", desc="Processing frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.copy())

        if prev_gray is not None:
            # compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                None,
                                                pyr_scale=0.5,
                                                levels=3,
                                                winsize=15,
                                                iterations=3,
                                                poly_n=5,
                                                poly_sigma=1.2,
                                                flags=0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            # mask out low-magnitude pixels (to ignore static background)
            mask = mag > args.mag_threshold
            if np.any(mask):
                mag_masked = mag[mask]
                ang_masked = ang[mask]
                angle_var = np.var(ang_masked)

                # rotation-aware motion score focused on moving parts of vid
                motion_score = np.mean(mag_masked) * (1 + args.angle_weight * angle_var)
            else:
                # no meaningful movement detected
                motion_score = 0.0

            motion_scores.append(motion_score)

        prev_gray = gray
        pbar.update(1)

    print(f"Processed {pbar.n} frames")
    cap.release()
    pbar.close()
    motion_scores = np.array(motion_scores)

    # smoothed motion scores
    smooth_scores = np.convolve(
        motion_scores, np.ones(args.smooth_kernel)/args.smooth_kernel, mode='same'
    )

    # break video into segments based on motion scores
    algo = rpt.Pelt(model="rbf").fit(smooth_scores)
    breakpoints = algo.predict(pen=args.penalty)

    # extract key frame from each segment (whichever frame has highest motion score)
    keyframe_indices = []
    start = 0

    for end in breakpoints:
        if end - start < args.min_seg_len:
            start = end
            continue

        segment = smooth_scores[start:end]
        key_idx = np.argmax(segment) + start
        keyframe_indices.append(key_idx)
        start = end

    for idx, frame_num in enumerate(keyframe_indices):
        frame_to_save = frames[frame_num+1]
        filename = os.path.join(args.output_dir, f"keyframe_{idx:03d}.jpg")
        cv2.imwrite(filename, frame_to_save)

    print(f"Saved {len(keyframe_indices)} keyframes to '{args.output_dir}'")

if __name__ == "__main__":
    main()