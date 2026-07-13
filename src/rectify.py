"""
Single-camera perspective rectification.

Per the project plan: rather than a calibrated multi-camera 3D fusion (which
would need a calibration grid we don't have, plus possible refractive
correction for the flat glass camera housings), each camera gets its own
simple 4-point perspective rectification so its view of the tank floor reads
as an approximate top-down rectangle. Two cameras per tank (facing each
other) each get rectified independently, then their heatmaps are combined by
flipping one camera's axis so "near the light wall" lines up consistently
between the two - see heatmap.py.

Corner points below were picked by eye from one sample frame per camera
(fractions of frame width/height, so they're resolution-independent). Only
`bottom_left` has been checked against a real frame so far - the other three
use a generic symmetric fallback until each camera's own sample frame gets
inspected. Re-check CALIBRATION_POINTS against real frames for each camera
during verification; a wrong trapezoid won't crash anything; it'll just place
heatmap density in the wrong spot.
"""

import cv2
import numpy as np

RECTIFIED_SIZE = (600, 800)  # (width, height) of the rectified output space

# Each entry: (near_left, near_right, far_left, far_right) as (x_frac, y_frac)
# "near" = bottom of frame (closest to this camera's own wall)
# "far"  = top of frame (toward the opposite wall / other camera)
CALIBRATION_POINTS = {
    "bottom_left": {
        "near_left": (0.0, 0.998),
        "near_right": (1.0, 0.998),
        "far_left": (0.30, 0.729),
        "far_right": (0.925, 0.622),
    },
    # Picked by eye from data/calibration/bottom_right_v2_sample.jpg.
    "bottom_right": {
        "near_left": (0.0, 0.998),
        "near_right": (1.0, 0.998),
        "far_left": (0.32, 0.62),
        "far_right": (0.88, 0.58),
    },
    # Picked by eye from data/calibration/top_left_sample.jpg (this is the
    # frame originally mis-copied as bottom_right - same trapezoid estimate,
    # just correctly attributed now).
    "top_left": {
        "near_left": (0.0, 0.998),
        "near_right": (1.0, 0.998),
        "far_left": (0.25, 0.60),
        "far_right": (0.78, 0.70),
    },
    # Picked by eye from data/calibration/top_right_sample.jpg.
    "top_right": {
        "near_left": (0.0, 0.998),
        "near_right": (1.0, 0.998),
        "far_left": (0.22, 0.42),
        "far_right": (0.75, 0.50),
    },
    # Generic fallback trapezoid for cameras not yet individually inspected -
    # a symmetric, slightly-conservative floor region. Replace once a real
    # sample frame from each camera has been checked.
    "_default": {
        "near_left": (0.0, 0.98),
        "near_right": (1.0, 0.98),
        "far_left": (0.30, 0.65),
        "far_right": (0.70, 0.65),
    },
}


def _get_camera_points(camera_name):
    return CALIBRATION_POINTS.get(camera_name, CALIBRATION_POINTS["_default"])


def get_homography(frame_w, frame_h, camera_name):
    """Returns the 3x3 homography matrix mapping this camera's raw pixel
    coordinates to the shared RECTIFIED_SIZE top-down space."""
    pts = _get_camera_points(camera_name)
    src = np.float32([
        [pts["near_left"][0] * frame_w, pts["near_left"][1] * frame_h],
        [pts["near_right"][0] * frame_w, pts["near_right"][1] * frame_h],
        [pts["far_right"][0] * frame_w, pts["far_right"][1] * frame_h],
        [pts["far_left"][0] * frame_w, pts["far_left"][1] * frame_h],
    ])
    w, h = RECTIFIED_SIZE
    # near edge -> bottom of rectified image, far edge -> top
    dst = np.float32([
        [0, h],
        [w, h],
        [w, 0],
        [0, 0],
    ])
    return cv2.getPerspectiveTransform(src, dst)


def rectify_frame(frame, camera_name):
    """Warp a full frame into the top-down rectified space (for visual QA)."""
    h_img, w_img = frame.shape[:2]
    H = get_homography(w_img, h_img, camera_name)
    return cv2.warpPerspective(frame, H, RECTIFIED_SIZE)


def rectify_points(points_xy, frame_w, frame_h, camera_name, flip_axis=False):
    """
    points_xy: list of (x, y) pixel coordinates (e.g. box-bottom-center of a
               detection - scallops sit on the floor, so the bottom-center of
               the box is a better floor-position estimate than the centroid).
    flip_axis: if True, flips the "near/far" axis so this camera's near wall
               maps to the *far* side of the shared rectified space - use
               this for whichever of the two cameras in a tank is NOT the
               one whose wall the light is on this session, so both cameras'
               heatmaps agree on which end is "near the light."
    Returns: list of (x, y) in RECTIFIED_SIZE coordinate space.
    """
    if not points_xy:
        return []
    H = get_homography(frame_w, frame_h, camera_name)
    pts = np.float32(points_xy).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    w, h = RECTIFIED_SIZE
    if flip_axis:
        warped = np.stack([w - warped[:, 0], h - warped[:, 1]], axis=1)

    return [tuple(p) for p in warped]
