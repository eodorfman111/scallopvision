"""
Sample frames from a set of videos and auto-label them with the current
best.pt model, writing YOLO-format image+txt pairs ready to drag into
Roboflow for review/correction and re-upload to the dataset.

Each source video already spans a wide range of real time (these are
multi-day timelapses - see pipeline.py's docstring), so evenly-spaced
sampling naturally yields diverse lighting/scene conditions per video
without any extra effort.

Usage:
    python export_roboflow_frames.py --out ../data/roboflow_export --per-video 25 \
        ../data/raw/bottom_left.mp4 ../data/raw/bottom_right.mp4 \
        ../data/raw/top_left.mp4 ../data/raw/top_right.mp4
"""

import argparse
import os

import cv2
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best.pt")
CONF_THRESHOLD = 0.25  # lower than the pipeline's 0.35 - err toward including borderline boxes for a human to review/correct in Roboflow


def scallop_class_index(model):
    """Resolve the 'scallop' class by name rather than assuming index 0 -
    some checkpoints carry stray non-scallop classes from upstream Roboflow
    annotation contamination (see pipeline.py's _scallop_class_index)."""
    for idx, name in model.names.items():
        if name == "scallop":
            return idx
    raise ValueError(f"No 'scallop' class found in model.names: {model.names}")


def sample_frame_indices(total_frames, n):
    if n >= total_frames:
        return list(range(total_frames))
    step = total_frames / n
    return [int(i * step) for i in range(n)]


def export_video(video_path, out_dir, n_frames, model, scallop_idx):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(total, n_frames))

    written = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            h, w = frame.shape[:2]
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device="cpu", classes=[scallop_idx])[0]

            image_name = f"{basename}_f{frame_idx:06d}.jpg"
            label_name = f"{basename}_f{frame_idx:06d}.txt"
            cv2.imwrite(os.path.join(out_dir, image_name), frame)

            lines = []
            if len(results.boxes):
                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(os.path.join(out_dir, label_name), "w") as f:
                f.write("\n".join(lines))

            written += 1
        frame_idx += 1

    cap.release()
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-video", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(MODEL_PATH)
    scallop_idx = scallop_class_index(model)

    total_written = 0
    for video_path in args.videos:
        n = export_video(video_path, args.out, args.per_video, model, scallop_idx)
        print(f"{video_path}: wrote {n} frame/label pairs")
        total_written += n

    print(f"\nTotal: {total_written} image+label pairs written to {args.out}")
