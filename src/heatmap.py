"""
Cumulative session heatmap.

Adapted from flockmetric/run_flockmetric.py's build_heatmap() - the original
rebuilt the heat grid fresh every frame (a snapshot of just that frame's
detections). Here the grid persists and accumulates across the whole
session, since the actual deliverable is "where did scallops spend time over
the full recording," not a single-frame snapshot.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from rectify import RECTIFIED_SIZE

HEATMAP_SIGMA = 18  # gaussian blur radius in rectified-space pixels


class HeatmapAccumulator:
    def __init__(self, size=RECTIFIED_SIZE):
        w, h = size
        self.size = size
        self.heat = np.zeros((h, w), dtype=np.float32)

    def add_points(self, points_xy):
        h, w = self.heat.shape
        for x, y in points_xy:
            xi, yi = int(x), int(y)
            if 0 <= yi < h and 0 <= xi < w:
                self.heat[yi, xi] += 1.0

    def render(self, alpha_background=None, near_label=None, far_label=None):
        """
        Returns a BGR image of the accumulated heat, framed as a labeled
        top-down floor plan rather than a bare colored blob - a tank-outline
        border, corner tick marks, and near/far wall labels (when given) make
        it legible as "this is the tank floor seen from above, camera A's
        wall at the bottom / camera B's wall at the top" rather than an
        unlabeled color gradient.
        """
        heat = gaussian_filter(self.heat, sigma=HEATMAP_SIGMA)
        if heat.max() > 0:
            heat = heat / heat.max()
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)

        if alpha_background is not None:
            bg = cv2.resize(alpha_background, self.size)
            colored = cv2.addWeighted(bg, 0.4, colored, 0.6, 0)

        w, h = self.size
        pad_top, pad_bottom, pad_side = 54, 54, 30
        canvas_h = h + pad_top + pad_bottom
        canvas_w = w + pad_side * 2
        canvas = np.full((canvas_h, canvas_w, 3), (26, 20, 15), dtype=np.uint8)  # dark navy-ish background (BGR)
        canvas[pad_top:pad_top + h, pad_side:pad_side + w] = colored

        # Tank-floor outline border, like a schematic/floor-plan frame.
        border_color = (210, 210, 210)
        cv2.rectangle(canvas, (pad_side, pad_top), (pad_side + w - 1, pad_top + h - 1), border_color, 2)

        # Corner tick marks reinforce "this is a measured plan," not decoration.
        tick = 14
        for cx, cy in [(pad_side, pad_top), (pad_side + w, pad_top),
                        (pad_side, pad_top + h), (pad_side + w, pad_top + h)]:
            dx = -tick if cx > pad_side + w / 2 else tick
            dy = -tick if cy > pad_top + h / 2 else tick
            cv2.line(canvas, (cx, cy), (cx + dx, cy), border_color, 2)
            cv2.line(canvas, (cx, cy), (cx, cy + dy), border_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "TOP-DOWN VIEW OF TANK FLOOR", (pad_side, 22),
                    font, 0.52, (225, 225, 225), 1, cv2.LINE_AA)

        # Bottom of the grid = near camera A's wall, top of the grid = near
        # camera B's wall (per rectify.py's fixed flip convention).
        if far_label:
            text = f"{far_label} wall (far)"
            (tw, _), _ = cv2.getTextSize(text, font, 0.5, 1)
            cv2.putText(canvas, text, (pad_side + (w - tw) // 2, pad_top - 10),
                        font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if near_label:
            text = f"{near_label} wall (near)"
            (tw, _), _ = cv2.getTextSize(text, font, 0.5, 1)
            cv2.putText(canvas, text, (pad_side + (w - tw) // 2, canvas_h - 16),
                        font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return canvas

    def half_fractions(self):
        """
        Fraction of total accumulated density in the bottom half vs. top half
        of the rectified space. Per rectify.py's convention, camera A's own
        wall maps to the bottom of the grid and camera B's (flipped) wall
        maps to the top - so this directly answers "what fraction of tracked
        time was spent near camera A's wall vs. camera B's wall," which -
        since each wall can carry its own light color - is the actual
        light-preference comparison for a session, not just a generic
        near/far split.
        """
        total = self.heat.sum()
        if total == 0:
            return 0.0, 0.0
        h = self.heat.shape[0]
        mid = h // 2
        near_a = self.heat[mid:, :].sum()   # bottom half = near camera A's wall
        near_b = self.heat[:mid, :].sum()   # top half = near camera B's wall
        return float(near_a / total), float(near_b / total)

    def total_observations(self):
        return float(self.heat.sum())
