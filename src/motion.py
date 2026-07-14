"""
Motion score: how much scallops are repositioning between sampled frames,
versus staying put.

IMPORTANT context (see pipeline.py's module docstring): the source footage
is a compressed multi-day timelapse, so consecutive sampled frames are NOT
evenly spaced in real time - the gap between samples varies. That means this
score is NOT a calibrated speed/velocity ("cm per second") - it's "how far
scallops' detected positions shifted between one sampled snapshot and the
next," aggregated across a session. Treat it as a relative/comparative
signal (does tank A show more repositioning than tank B, does it change
under different light colors), not an absolute physical measurement.

No persistent tracking is attempted (rejected earlier for the same reason -
frames can be hours/days apart, so there's no such thing as "the same
scallop's continuous trajectory" here). Instead each consecutive frame PAIR
is matched independently with simple greedy nearest-neighbor matching in
rectified floor coordinates, capped at a maximum match distance so a
scallop on one side of the tank never gets "matched" to an unrelated one on
the far side just because both are the closest available point.
"""

import math

from rectify import RECTIFIED_SIZE

# A match beyond this distance (in rectified-space pixels) is almost
# certainly two different scallops, not the same one moving - cap it so
# those don't get counted as "movement."
MAX_MATCH_DISTANCE = min(RECTIFIED_SIZE) * 0.25


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _match_frame_pair(points_a, points_b, max_dist=MAX_MATCH_DISTANCE):
    """Greedy nearest-neighbor matching between two point sets. Returns a
    list of per-match distances (one per matched pair, unmatched points on
    either side are simply dropped - a scallop that appeared/disappeared
    between frames isn't "movement," it's a detection change)."""
    if not points_a or not points_b:
        return []

    candidates = []
    for i, pa in enumerate(points_a):
        for j, pb in enumerate(points_b):
            d = _dist(pa, pb)
            if d <= max_dist:
                candidates.append((d, i, j))
    candidates.sort(key=lambda c: c[0])

    used_a, used_b = set(), set()
    matched_distances = []
    for d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched_distances.append(d)
    return matched_distances


def compute_motion_score(frame_sequence):
    """
    frame_sequence: chronologically-ordered list of (light_color, rectified_points)
    for ONE camera - points already in shared rectified floor coordinates.

    Returns dict:
      overall_avg_shift: average per-scallop displacement between consecutive
        frames, across the whole session (rectified-space pixels; relative
        units, not a calibrated real-world distance)
      frame_pairs_matched: how many consecutive-frame pairs contributed at
        least one matched scallop (i.e. had usable signal)
      avg_shift_by_color: same average, broken out by the LATER frame's light
        color in each pair (so "motion score under green" etc.)
    """
    all_distances = []
    by_color_distances = {}
    pairs_with_signal = 0

    for (_, points_a), (color_b, points_b) in zip(frame_sequence, frame_sequence[1:]):
        matched = _match_frame_pair(points_a, points_b)
        if not matched:
            continue
        pairs_with_signal += 1
        pair_avg = sum(matched) / len(matched)
        all_distances.append(pair_avg)
        by_color_distances.setdefault(color_b, []).append(pair_avg)

    overall_avg = sum(all_distances) / len(all_distances) if all_distances else 0.0
    avg_by_color = {
        c: sum(vals) / len(vals) for c, vals in by_color_distances.items()
    }

    return {
        "overall_avg_shift": overall_avg,
        "frame_pairs_matched": pairs_with_signal,
        "avg_shift_by_color": avg_by_color,
    }
