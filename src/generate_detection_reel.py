"""
Generates a short annotated clip showing the detector running frame-by-frame
on real footage - boxes + confidence drawn on every frame - so a viewer can
see the model actually working, not just read a heatmap.

Downscaled and re-encoded to H.264 so it's both small enough to commit to a
public repo and playable in any browser via <video>/st.video().

Usage:
    python generate_detection_reel.py --video ../data/raw/bottom_left.mp4 \
        --start-frame 480 --num-frames 40 --out ../data/reels/bottom_reel.mp4
"""

import argparse
import os
import subprocess
import tempfile

import cv2
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best.pt")
CONF_THRESHOLD = 0.20  # matches pipeline.py - 0.30 was visibly missing overlapping scallops in dense clusters
INFERENCE_IMGSZ = 1280  # matches pipeline.py - the model was trained at this resolution
BATCH_SIZE = 15  # chunk predict() calls rather than one giant batch, same reasoning as pipeline.py
OUTPUT_SIZE = (1280, 720)
BOX_COLOR = (60, 220, 130)   # BGR - a bright green that reads clearly on any tank lighting
BOX_THICKNESS = 3


def scallop_class_index(model):
    for idx, name in model.names.items():
        if name == "scallop":
            return idx
    raise ValueError(f"No 'scallop' class found in model.names: {model.names}")


def generate_reel(video_path, start_frame, num_frames, out_path, fps_override=None):
    model = YOLO(MODEL_PATH)
    scallop_idx = scallop_class_index(model)

    cap = cv2.VideoCapture(video_path)
    src_fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 10.0

    frames = []
    frame_idx = 0
    while frame_idx < start_frame + num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= start_frame:
            frames.append(frame)
        frame_idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path} at start_frame={start_frame}")

    results_list = []
    for batch_start in range(0, len(frames), BATCH_SIZE):
        batch = frames[batch_start:batch_start + BATCH_SIZE]
        results_list.extend(model.predict(batch, conf=CONF_THRESHOLD, verbose=False, device="cpu",
                                            imgsz=INFERENCE_IMGSZ, classes=[scallop_idx]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = os.path.join(tmp_dir, "raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_path, fourcc, src_fps, OUTPUT_SIZE)

        total_detections = 0
        for frame, results in zip(frames, results_list):
            frame = cv2.resize(frame, OUTPUT_SIZE)
            scale_x = OUTPUT_SIZE[0] / frames[0].shape[1]
            scale_y = OUTPUT_SIZE[1] / frames[0].shape[0]

            if len(results.boxes):
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                total_detections += len(boxes)
                for (x1, y1, x2, y2), conf in zip(boxes, confs):
                    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                    label = f"scallop {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), BOX_COLOR, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

            writer.write(frame)
        writer.release()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Re-encode to H.264 (yuv420p) - mp4v output from cv2 isn't reliably
        # playable in browsers; libx264 is universally supported.
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", out_path],
            check=True, capture_output=True,
        )

    print(f"Wrote {out_path} ({len(frames)} frames, {total_detections} total detections, "
          f"avg {total_detections / len(frames):.1f}/frame)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    generate_reel(args.video, args.start_frame, args.num_frames, args.out, args.fps)
