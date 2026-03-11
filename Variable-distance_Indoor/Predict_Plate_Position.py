import os
import cv2
import json
import numpy as np
from math import sqrt
from joblib import Parallel, delayed
from utils.rle_utils import rle_decode
import csv
import pandas as pd
import argparse

# Default root
DEFAULT_VIDEO_FRAMES_ROOT = "./indoor_video_frames"   # contains P{pid}/0frame, 500frame, ...
DEFAULT_GSAM_OUTPUT_ROOT  = "./indoor_gaze_outputs"   # contains P{pid}/0frame/{mask_data,json_data,result}

RAW_DATA_ROOT = "./GazeDepth/Variable-distance_viewing/Variable-distance_indoor/Timeseries_Data_and_Scene_Video"

DEFAULT_PREDICT_PLATE_OUTPUT_ROOT = "./predict_plate"        # will create predict_plate/P{pid}/

PARTICIPANTS = range(1, 20)

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--video_frames_root", type=str, default=DEFAULT_VIDEO_FRAMES_ROOT,
                    help="Root folder containing extracted frames (expects P{pid}/ subfolders)")
parser.add_argument("--gsam_output_root", type=str, default=DEFAULT_GSAM_OUTPUT_ROOT,
                    help="Root folder containing GSAM outputs (expects P{pid}/ subfolders)")
parser.add_argument("--predict_plate_output_root", type=str, default=DEFAULT_PREDICT_PLATE_OUTPUT_ROOT,
                    help="Root folder to save predicted plate overlays/CSVs")
args = parser.parse_args()

VIDEO_FRAMES_ROOT = args.video_frames_root
GSAM_OUTPUT_ROOT  = args.gsam_output_root
PREDICT_PLATE_ROOT = args.predict_plate_output_root

def find_subject_folder(raw_root: str, pid: int) -> str | None:
    # Example folder name:
    # variable-distance_indoor_p1_2024-08-31_10-55-08_-16abf83e
    token = f"_p{pid}_"
    for name in os.listdir(raw_root):
        full = os.path.join(raw_root, name)
        if token in name and os.path.isdir(full):
            return full
    return None

def is_zero_box(obj):
    return all(obj[k] == 0 for k in ["x1", "y1", "x2", "y2"])

def get_center_bottom(obj):
    cx = (obj["x1"] + obj["x2"]) / 2
    cy = obj["y2"]
    return cx, cy

def get_center(obj):
    cx = (obj["x1"] + obj["x2"]) / 2
    cy = (obj["y1"] + obj["y2"]) / 2
    return cx, cy

def get_table_slope(mask_path, json_path):
    with open(mask_path, 'r') as f:
        rle_json = json.load(f)
    mask = rle_decode(rle_json['rle'], rle_json['shape'])
    with open(json_path, 'r') as f:
        data = json.load(f)

    table_coords = []
    for obj_id, obj in data['labels'].items():
        if obj['class_name'] in ["table", "brown", "brown table"]:
            table_mask = (mask == int(obj_id))
            y_idx, x_idx = np.where(table_mask)
            coords = list(zip(x_idx, y_idx))
            table_coords.extend(coords)

    if len(table_coords) < 2:
        return 0.0
    x_all, y_all = zip(*table_coords)
    slope = np.polyfit(x_all, y_all, 1)[0]
    return slope

def process_distance(folder):
    distances = []
    json_dir = os.path.join(base_dir, folder, "json_data")
    mask_dir = os.path.join(base_dir, folder, "mask_data")
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    for json_file in json_files:
        frame_id = json_file.split("_")[-1].split(".")[0]
        json_path = os.path.join(json_dir, json_file)
        mask_path = os.path.join(mask_dir, f"mask_{frame_id}.npy")

        with open(json_path, 'r') as f:
            data = json.load(f)

        objs = data['labels'].values()
        plates = [o for o in objs if o['class_name'] == 'plate' and not is_zero_box(o)]
        screens = [o for o in objs if o['class_name'] == 'screen' and not is_zero_box(o)]
        filtered_plates = [p for p in plates if not any(s['x1'] <= get_center(p)[0] <= s['x2'] and s['y1'] <= get_center(p)[1] <= s['y2'] for s in screens)]
        plates = filtered_plates

        ladies = [o for o in objs if o['class_name'] == 'lady' and not is_zero_box(o)]
        filtered_ladies = [l for l in ladies if not any((l['y1'] + l['y2']) / 2 > (p['y1'] + p['y2']) / 2 for p in plates)]
        ladies = filtered_ladies

        if len(ladies) >= 1 and len(plates) >= 1:
            lady = max(ladies, key=lambda o: o['x1'])
            lx, ly = get_center_bottom(lady)
            plate = min(
                plates,
                key=lambda p: sqrt((get_center(p)[0] - lx)**2 + (get_center(p)[1] - ly)**2)
            )
            px, py = get_center(plate)
            dist = sqrt((lx - px) ** 2 + (ly - py) ** 2)
            distances.append(dist)

    return distances

def find_nearest_gaze(gaze_df, target_ns, max_diff_ns=500_000_000):
    diffs = np.abs(gaze_df["timestamp [ns]"] - target_ns)
    idx = diffs.idxmin()
    if diffs.iloc[idx] > max_diff_ns:
        return None
    return gaze_df.iloc[idx]

def process_prediction(folder, avg_dist):
    results = []
    json_dir = os.path.join(base_dir, folder, "json_data")
    mask_dir = os.path.join(base_dir, folder, "mask_data")
    image_dir = os.path.join(video_dir, folder)
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    for json_file in json_files:
        frame_id = json_file.split("_")[-1].split(".")[0]
        frame_idx = int(frame_id)

        if frame_idx >= len(world_ts):
            continue
        target_ns = world_ts[frame_idx]
        gaze_row = find_nearest_gaze(gaze_df, target_ns)
        if gaze_row is None:
            continue
        gaze_x = int(gaze_row["gaze x [px]"])
        gaze_y = int(gaze_row["gaze y [px]"])

        json_path = os.path.join(json_dir, json_file)
        mask_path = os.path.join(mask_dir, f"mask_{frame_id}.npy")
        image_path = os.path.join(image_dir, f"{frame_id}.jpg")

        with open(json_path, 'r') as f:
            data = json.load(f)

        objs = data['labels'].values()
        plates = [o for o in objs if o['class_name'] == 'plate' and not is_zero_box(o)]
        screens = [o for o in objs if o['class_name'] == 'screen' and not is_zero_box(o)]
        filtered_plates = [p for p in plates if not any(s['x1'] <= get_center(p)[0] <= s['x2'] and s['y1'] <= get_center(p)[1] <= s['y2'] for s in screens)]
        plates = filtered_plates

        ladies = [o for o in objs if o['class_name'] == 'lady' and not is_zero_box(o)]
        filtered_ladies = [l for l in ladies if not any((l['y1'] + l['y2']) / 2 > (p['y1'] + p['y2']) / 2 for p in plates)]
        ladies = filtered_ladies

        screen_x = None
        screen_objs = [o for o in objs if o['class_name'] == 'screen' and not is_zero_box(o)]
        if len(screen_objs) > 0:
            screen = max(screen_objs, key=lambda o: o['x1'])
            screen_x = (screen['x1'] + screen['x2']) / 2

        if len(ladies) >= 1 and len(plates) == 0:
            image = cv2.imread(image_path)
            slope = get_table_slope(mask_path, json_path)
            lady = max(ladies, key=lambda o: o['x1'])
            lx, ly = get_center_bottom(lady)

            if abs(slope) < 1e-3:
                dx, dy = 0.0, -1.0
            else:
                dx, dy = 1.0, -1.0 / slope
            norm = np.sqrt(dx ** 2 + dy ** 2)
            dx, dy = dx / norm, dy / norm
            if dy < 0:
                dx *= -1
                dy *= -1

            px = int(lx + dx * avg_dist)
            py = int(ly + dy * avg_dist)
            h, w = image.shape[:2]

            if 0 <= px < w and 0 <= py < h:
                cv2.circle(image, (px, py), 8, (255, 0, 0), -1)
                intercept_y_plate = py - slope * px
                start_y = int(slope * 0 + intercept_y_plate)
                end_y = int(slope * w + intercept_y_plate)
                cv2.line(image, (0, start_y), (w, end_y), (255, 0, 0), 2)

                lady_x = (lady['x1'] + lady['x2']) / 2
                lady_y = lady['y1']
                lady_x1 = lady['x1']
                lady_x2 = lady['x2']
                intercept_y_div = lady_y - slope * lady_x
                start_y_div = int(slope * 0 + intercept_y_div)
                end_y_div = int(slope * w + intercept_y_div)
                cv2.line(image, (0, start_y_div), (w, end_y_div), (0, 255, 0), 2)

                cv2.circle(image, (gaze_x, gaze_y), 10, (0, 0, 255), -1)

                y_at_plate = int(slope * gaze_x + intercept_y_plate)
                y_at_div = int(slope * gaze_x + intercept_y_div)

                label = ""

                if intercept_y_plate is not None and gaze_y > y_at_plate:
                    label = "near"

                elif (
                    lady_x1 is not None and lady_x2 is not None and
                    lady_x1 <= gaze_x <= lady_x2 and
                    (
                        (intercept_y_plate is None and intercept_y_div is not None and gaze_y >= y_at_div)
                        or
                        (intercept_y_plate is not None and intercept_y_div is not None and
                        y_at_plate >= gaze_y >= y_at_div)
                    )
                ):
                    label = "middle"

                elif (
                    (
                        lady_x1 is not None and lady_x2 is not None and
                        screen_x is None and
                        gaze_y is not None and intercept_y_div is not None and
                        gaze_y < y_at_div
                    )
                    or
                    (
                        intercept_y_div is not None and gaze_y < y_at_div
                    )
                ):
                    label = "far"

                results.append([frame_id, label])

            output_path = os.path.join(output_dir, f"{folder}_{frame_id}.jpg")
            cv2.imwrite(output_path, image)

    return results

for pid in PARTICIPANTS:
    print(f"\n===== Processing P{pid} =====")

    video_dir = f"{VIDEO_FRAMES_ROOT}/P{pid}"
    base_dir  = f"{GSAM_OUTPUT_ROOT}/P{pid}"

    if not os.path.isdir(base_dir):
        print(f"[WARN] base_dir not found, skip P{pid}: {base_dir}")
        continue
    if not os.path.isdir(video_dir):
        print(f"[WARN] video_dir not found, skip P{pid}: {video_dir}")
        continue

    subj_raw_dir = find_subject_folder(RAW_DATA_ROOT, pid)
    if subj_raw_dir is None:
        print(f"[WARN] raw subject folder not found for P{pid}")
        continue

    gaze_csv_path   = os.path.join(subj_raw_dir, "gaze.csv")
    world_ts_path   = os.path.join(subj_raw_dir, "world_timestamps.csv")

    output_dir = os.path.join(PREDICT_PLATE_ROOT, f"P{pid}")
    output_csv = os.path.join(output_dir, "predicted_plate_labels.csv")

    os.makedirs(output_dir, exist_ok=True)
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and "frame" in f]

    gaze_df = pd.read_csv(gaze_csv_path)
    world_df = pd.read_csv(world_ts_path)
    world_ts = world_df["timestamp [ns]"].to_numpy()

    distance_lists = Parallel(n_jobs=-1)(delayed(process_distance)(folder) for folder in subfolders)
    distances = [d for lst in distance_lists for d in lst]
    avg_dist = np.mean(distances)
    print(f"Average distance: {avg_dist:.2f} pixels")

    results = Parallel(n_jobs=-1)(delayed(process_prediction)(folder, avg_dist) for folder in subfolders)
    all_results = [r for sublist in results for r in sublist]

    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "auto"])
        writer.writerows(all_results)

    print(base_dir)
    print(f"{pid} done: saved predicted plate locations and gaze labels")
