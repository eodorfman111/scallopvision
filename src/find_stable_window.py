"""
Scans a video for the most visually stable ~N-frame window (lowest frame-to-
frame pixel change), so detection reels don't land on a timelapse jump-cut
(sudden lighting/color change between consecutive frames, since consecutive
frames in this footage can be minutes-to-hours apart in real time).

Usage:
    python find_stable_window.py --video ../data/raw/bottom_left.mp4 --window 40
"""

import argparse

import cv2
import numpy as np


def scan(video_path, window, stride=10, sample_size=(160, 90)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    small_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small_frames.append(cv2.resize(frame, sample_size).astype(np.float32))
        frame_idx += 1
    cap.release()

    n = len(small_frames)
    diffs = [np.abs(small_frames[i] - small_frames[i - 1]).mean() for i in range(1, n)]

    best_start, best_score = 0, float("inf")
    for start in range(0, n - window, stride):
        window_diffs = diffs[start:start + window - 1]
        score = max(window_diffs) if window_diffs else float("inf")  # worst single jump in the window
        if score < best_score:
            best_score = score
            best_start = start

    print(f"{video_path}: {n} frames total, best window start={best_start} "
          f"(max frame-to-frame diff={best_score:.2f})")
    return best_start


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    scan(args.video, args.window, args.stride)
