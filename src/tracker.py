"""
Greedy IoU-matching centroid tracker.
Ported from flockmetric/run_roboflow.py's CentroidTracker (weight-estimation
logic dropped — not relevant to scallops).
"""

import numpy as np

MAX_LOST = 8          # frames a track can go unmatched before being dropped
IOU_MATCH_THRESHOLD = 0.15


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua


class Track:
    _next_id = 1

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box          # (x1, y1, x2, y2)
        self.lost = 0


class CentroidTracker:
    def __init__(self):
        self.tracks: list[Track] = []

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2)
        returns:    list of (track_id, x1, y1, x2, y2)
        """
        matched_det = set()
        matched_trk = set()

        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for ti, trk in enumerate(self.tracks):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = iou(trk.box, det)

            while True:
                ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[ti, di] < IOU_MATCH_THRESHOLD:
                    break
                if ti not in matched_trk and di not in matched_det:
                    self.tracks[ti].box = detections[di]
                    self.tracks[ti].lost = 0
                    matched_trk.add(ti)
                    matched_det.add(di)
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

        for ti, trk in enumerate(self.tracks):
            if ti not in matched_trk:
                trk.lost += 1

        for di, det in enumerate(detections):
            if di not in matched_det:
                self.tracks.append(Track(det))

        self.tracks = [t for t in self.tracks if t.lost <= MAX_LOST]

        results = []
        for trk in self.tracks:
            if trk.lost > 0:
                continue
            x1, y1, x2, y2 = trk.box
            results.append((trk.id, x1, y1, x2, y2))

        return results
