"""
Per-frame light-state classifier.

No reliable external schedule/log of light color or day/night exists for this
client's sessions (confirmed by CFF - batteries died unpredictably, and the
light *position* rotation wasn't strictly logged either). So light color and
day/night state must be read directly off the footage, not assumed from a
calendar.

Approach: the whole tank tints with whichever "fishing light" color is active
(water scattering spreads the color across the frame, not just one wall - see
project notes), so a simple dominant-hue read on saturated pixels is enough to
tell green vs. blue apart. Per the client's own explanation, red light stands
in for "night" (scallops can't see red, so it's functionally dark for them),
so red-dominant / very-dark frames are classified as night; anything else
(green or blue clearly present) is classified as day.

These hue ranges were picked from visual inspection of a handful of sample
frames, not a calibrated color chart - re-check against real footage during
verification and adjust HUE_RANGES if a session gets misclassified.
"""

import cv2
import numpy as np

# OpenCV hue is 0-179 (half of the usual 0-360 degree wheel)
HUE_RANGES = {
    "red": [(0, 12), (168, 180)],   # wraps around 0
    "green": [(35, 95)],
    "blue": [(96, 145)],
}

MIN_SATURATION = 40     # ignore near-gray/white pixels (glare, sediment) when reading hue
MIN_VALUE = 20          # ignore near-black pixels (shadow) too
DARK_FRAME_MEAN_V = 35  # below this, treat as "dark" regardless of hue -> night


def _hue_in_ranges(hue, ranges):
    return any(lo <= hue < hi for lo, hi in ranges)


def classify_frame(frame_bgr):
    """
    Returns (light_color, day_night) where:
      light_color in {"green", "blue", "red", "unknown"}
      day_night   in {"day", "night"}
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask = (s >= MIN_SATURATION) & (v >= MIN_VALUE)
    mean_v = float(v.mean())

    if mask.sum() < 0.01 * mask.size:
        # Almost nothing saturated enough to read a color from - call it dark/night
        return "unknown", "night"

    hue_pixels = h[mask]
    # Histogram over the saturated pixels, pick the color bucket with the most votes
    counts = {}
    for name, ranges in HUE_RANGES.items():
        counts[name] = sum(_hue_in_ranges(int(hh), ranges) for hh in hue_pixels[::37])  # subsample for speed

    light_color = max(counts, key=counts.get) if any(counts.values()) else "unknown"

    if light_color == "red" or mean_v < DARK_FRAME_MEAN_V:
        day_night = "night"
    else:
        day_night = "day"

    return light_color, day_night


def classify_session(video_path, sample_every_n_frames=30, max_samples=60):
    """
    Classify an entire session by sampling frames throughout the video and
    taking the majority vote - more robust than trusting a single frame,
    since brief transients (e.g. someone's flashlight, a reflection) shouldn't
    flip the whole session's label.

    Returns (light_color, day_night, vote_counts) where vote_counts is a dict
    of {(color, day_night): count} for transparency/debugging.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    votes = {}
    sampled = 0
    frame_idx = 0

    while sampled < max_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        color, day_night = classify_frame(frame)
        key = (color, day_night)
        votes[key] = votes.get(key, 0) + 1
        sampled += 1
        frame_idx += sample_every_n_frames
        if frame_idx >= total:
            break

    cap.release()

    if not votes:
        return "unknown", "night", {}

    (best_color, best_day_night), _ = max(votes.items(), key=lambda kv: kv[1])
    return best_color, best_day_night, votes
