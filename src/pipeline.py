"""
Orchestrates a full tank analysis: for each sampled frame from each camera,
independently classify that frame's own light color/day-night, detect
scallops, rectify their floor positions, and accumulate a heatmap bucketed
by light color across the whole recording.

IMPORTANT finding from real footage (2026-07-12): the source videos are not
continuous single sessions - they are multi-day timelapse compilations
(consecutive frames can be hours to days apart in real time, confirmed via
burned-in camera timestamps jumping non-monotonically in small frame steps).
That makes frame-to-frame scallop tracking meaningless (a "track" between two
frames days apart isn't the same animal moving, it's an unrelated snapshot),
so this pipeline does NOT track individual scallops or compute session dwell
time. Instead each sampled frame is treated as an independent observation:
its own light color is classified on the spot (light color visibly changes
across the timelapse, consistent with the known fact that the light position
rotates through slots over time) and its detections are binned into a
heatmap keyed by that frame's light color. Aggregating over many days this
way is arguably a *better* light-preference signal than a single short
session would have been, since it naturally samples many rotations of the
light and many day/night cycles.

Usage:
    python pipeline.py --tank bottom \
        --cam-a data/raw/bottom_left.mp4 --cam-a-name bottom_left \
        --cam-b data/raw/bottom_right.mp4 --cam-b-name bottom_right \
        --out data/demo_sessions/bottom_session1
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO

from light_classifier import classify_frame
from rectify import rectify_points
from heatmap import HeatmapAccumulator

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best.pt")
CONF_THRESHOLD = 0.20  # 0.35 was visibly missing partially-overlapping scallops in dense clusters (confirmed by spot-checking boxes at multiple thresholds on real frames) - 0.20 catches those without adding obvious false positives
TARGET_SAMPLES_PER_CAMERA = 150  # each frame is already an independent, widely-spaced timelapse sample - 150 evenly-spaced samples gives solid coverage across the ~12-day span without an hour-long CPU run
INFERENCE_IMGSZ = 1280  # matches the resolution the new model was actually trained at (imgsz=1280 in the Colab run) - 320/640 both undercounted since the model's learned features are tuned for this input size
BATCH_SIZE = 20  # batch predict() calls instead of one-call-per-frame - Ultralytics' per-call overhead dominates runtime when called thousands of times individually


def _box_bottom_center(x1, y1, x2, y2):
    """Bottom-center of the box - a better floor-position estimate for a
    benthic animal than the box centroid, since the box's vertical extent
    mostly reflects the shell's height off the substrate near the camera."""
    return ((x1 + x2) / 2.0, y2)


def _scallop_class_index(model):
    """Some checkpoints carry stray non-scallop classes (e.g. '4', '44') from
    upstream annotation contamination in Roboflow - see the corrupt-label
    issue discussed with the client. Rather than assume class 0, resolve the
    "scallop" class by name so detections of any stray classes are excluded
    at inference time regardless of index."""
    for idx, name in model.names.items():
        if name == "scallop":
            return idx
    raise ValueError(f"No 'scallop' class found in model.names: {model.names}")


def process_camera(video_path, camera_name, flip_axis, model, scallop_class_idx):
    """
    Runs per-frame detection + light classification over one camera's video.
    No tracking across frames - each sampled frame is independent.

    Returns dict with:
      frames_sampled: int
      light_color_frame_counts: {color: n}
      day_night_frame_counts: {"day"/"night": n}
      light_color_observation_counts: {color: n}  (total detections seen under that color)
      rectified_points_by_color: {color: [(x, y), ...]}  (all detections' floor points, in shared rectified space)
    """
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_samples = min(TARGET_SAMPLES_PER_CAMERA, total_frames)
    stride = max(1, total_frames // n_samples)

    # Sequential decode + stride-skip, NOT seeking - cap.set(POS_FRAMES) was
    # benchmarked far slower than just reading forward on this footage's
    # encoding (~0.7s/seek vs ~0.03s/frame sequential), almost certainly
    # because seeking has to decode forward from the nearest keyframe anyway
    # on a long-GOP stream. Sequential read-and-skip is much faster in practice.
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()

    light_color_frame_counts = defaultdict(int)
    day_night_frame_counts = defaultdict(int)
    light_color_observation_counts = defaultdict(int)
    rectified_points_by_color = defaultdict(list)

    # Classify on a downsized copy - a global dominant-hue read doesn't need
    # full 4K resolution, and this cut per-frame classification time enormously.
    frame_colors = []
    for frame in frames:
        small = cv2.resize(frame, (320, 180))
        light_color, day_night = classify_frame(small)
        frame_colors.append(light_color)
        light_color_frame_counts[light_color] += 1
        day_night_frame_counts[day_night] += 1

    # Batch the detector calls - calling .predict() once per frame in a tight
    # loop carries large per-call overhead in Ultralytics that dominates
    # runtime at these frame counts; batching collapses that overhead.
    for batch_start in range(0, len(frames), BATCH_SIZE):
        batch = frames[batch_start:batch_start + BATCH_SIZE]
        batch_colors = frame_colors[batch_start:batch_start + BATCH_SIZE]
        results_list = model.predict(batch, conf=CONF_THRESHOLD, verbose=False, device="cpu",
                                       imgsz=INFERENCE_IMGSZ, classes=[scallop_class_idx])
        for results, light_color in zip(results_list, batch_colors):
            if len(results.boxes):
                raw_points = [_box_bottom_center(*b) for b in results.boxes.xyxy.cpu().numpy()]
                rectified = rectify_points(raw_points, frame_w, frame_h, camera_name, flip_axis=flip_axis)
                rectified_points_by_color[light_color].extend(rectified)
                light_color_observation_counts[light_color] += len(rectified)

    return {
        "frames_sampled": len(frames),
        "light_color_frame_counts": dict(light_color_frame_counts),
        "day_night_frame_counts": dict(day_night_frame_counts),
        "light_color_observation_counts": dict(light_color_observation_counts),
        "rectified_points_by_color": dict(rectified_points_by_color),
    }


def run_session(tank_name, cam_a_path, cam_a_name, cam_b_path, cam_b_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(MODEL_PATH)
    scallop_idx = _scallop_class_index(model)

    print(f"[{tank_name}] Processing {cam_a_name} (per-frame, no tracking)...")
    res_a = process_camera(cam_a_path, cam_a_name, flip_axis=False, model=model, scallop_class_idx=scallop_idx)
    print(f"  -> {res_a['frames_sampled']} frames sampled, "
          f"colors seen: {res_a['light_color_frame_counts']}")

    print(f"[{tank_name}] Processing {cam_b_name} (per-frame, no tracking)...")
    res_b = process_camera(cam_b_path, cam_b_name, flip_axis=True, model=model, scallop_class_idx=scallop_idx)
    print(f"  -> {res_b['frames_sampled']} frames sampled, "
          f"colors seen: {res_b['light_color_frame_counts']}")

    # Merge both cameras' per-color rectified points into one accumulator per color.
    all_colors = set(res_a["rectified_points_by_color"]) | set(res_b["rectified_points_by_color"])
    heatmaps = {}
    half_fractions_by_color = {}
    for color in all_colors:
        acc = HeatmapAccumulator()
        acc.add_points(res_a["rectified_points_by_color"].get(color, []))
        acc.add_points(res_b["rectified_points_by_color"].get(color, []))
        heatmaps[color] = acc
        half_fractions_by_color[color] = acc.half_fractions()

    # Store absolute paths - stats.json may later be read by a process (e.g.
    # the Streamlit app) running from a different working directory than
    # this pipeline script.
    heatmap_paths = {}
    for color, acc in heatmaps.items():
        path = os.path.abspath(os.path.join(out_dir, f"heatmap_{color}.png"))
        cv2.imwrite(path, acc.render(near_label=cam_a_name, far_label=cam_b_name))
        heatmap_paths[color] = path

    # Combined "overall" heatmap across all colors, for a single at-a-glance image.
    overall_acc = HeatmapAccumulator()
    for acc in heatmaps.values():
        overall_acc.heat += acc.heat
    overall_path = os.path.abspath(os.path.join(out_dir, "heatmap_overall.png"))
    cv2.imwrite(overall_path, overall_acc.render(near_label=cam_a_name, far_label=cam_b_name))
    heatmap_paths["overall"] = overall_path

    def _merge_counts(key):
        merged = defaultdict(int)
        for res in (res_a, res_b):
            for k, v in res[key].items():
                merged[k] += v
        return dict(merged)

    light_color_observation_counts = _merge_counts("light_color_observation_counts")
    light_color_frame_counts_merged = _merge_counts("light_color_frame_counts")
    total_obs = sum(light_color_observation_counts.values())
    light_color_observation_fractions = (
        {c: n / total_obs for c, n in light_color_observation_counts.items()} if total_obs else {}
    )

    # The scallop population is fixed and enclosed - it doesn't grow or shrink
    # between frames. So raw observation totals/fractions are NOT a fair
    # comparison across colors: a color that happened to get sampled in more
    # frames will accumulate a bigger total even at an identical per-frame
    # rate. The scientifically valid comparison is average detections per
    # frame OF THAT COLOR - this is the number that actually reflects
    # "were more scallops visible/positioned favorably when this color was
    # active," independent of how many frames of each color got sampled.
    light_color_avg_per_frame = {
        c: light_color_observation_counts[c] / light_color_frame_counts_merged[c]
        for c in light_color_observation_counts
        if light_color_frame_counts_merged.get(c)
    }

    stats = {
        "tank": tank_name,
        "camera_a": {
            "name": cam_a_name,
            "frames_sampled": res_a["frames_sampled"],
            "light_color_frame_counts": res_a["light_color_frame_counts"],
            "day_night_frame_counts": res_a["day_night_frame_counts"],
            "observation_count": sum(res_a["light_color_observation_counts"].values()),
        },
        "camera_b": {
            "name": cam_b_name,
            "frames_sampled": res_b["frames_sampled"],
            "light_color_frame_counts": res_b["light_color_frame_counts"],
            "day_night_frame_counts": res_b["day_night_frame_counts"],
            "observation_count": sum(res_b["light_color_observation_counts"].values()),
        },
        "light_color_observation_counts": light_color_observation_counts,
        "light_color_observation_fractions": light_color_observation_fractions,
        "light_color_avg_per_frame": light_color_avg_per_frame,
        "half_fractions_by_color": half_fractions_by_color,
        "total_floor_observations": total_obs,
        "heatmap_paths": heatmap_paths,
    }

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone. Wrote {len(heatmap_paths)} heatmaps and {stats_path}")
    print(json.dumps({k: v for k, v in stats.items() if k != "heatmap_paths"}, indent=2))
    return stats, heatmap_paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tank", required=True)
    ap.add_argument("--cam-a", required=True)
    ap.add_argument("--cam-a-name", required=True)
    ap.add_argument("--cam-b", required=True)
    ap.add_argument("--cam-b-name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_session(args.tank, args.cam_a, args.cam_a_name, args.cam_b, args.cam_b_name, args.out)
