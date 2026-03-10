import os
import cv2
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import csv
import torch
import gc
from utils.rle_utils import rle_decode
import argparse

# Default root
DEFAULT_VIDEO_FRAMES_ROOT = "./indoor_video_frames"   # contains P{pid}/0frame, 500frame, ...
DEFAULT_GSAM_OUTPUT_ROOT = "./indoor_gaze_outputs"   # contains P{pid}/0frame/{mask_data,json_data,result}

RAW_DATA_ROOT = "./GazeDepth/Variable-distance_viewing/Variable-distance_indoor/Timeseries_Data_and_Scene_Video"

PARTICIPANTS = range(1, 20)  # P1 ~ P19

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--video_frames_root", type=str, default=DEFAULT_VIDEO_FRAMES_ROOT,
                    help="Root folder containing extracted frames (expects P{pid}/ subfolders)")
parser.add_argument("--gsam_output_root", type=str, default=DEFAULT_GSAM_OUTPUT_ROOT,
                    help="Root folder containing GSAM outputs (expects P{pid}/ subfolders)")
args = parser.parse_args()

VIDEO_FRAMES_ROOT = args.video_frames_root
GSAM_OUTPUT_ROOT = args.gsam_output_root

def find_nearest_gaze(gaze_df: pd.DataFrame, target_ns: int):
    diffs = np.abs(gaze_df["timestamp [ns]"] - target_ns)
    idx = diffs.idxmin()
    if diffs.loc[idx] > 500_000_000:
        return None
    return gaze_df.loc[idx]


def find_subject_folder(raw_root: str, pid: int) -> str | None:
    # Example folder name:
    # variable-distance_indoor_p1_2024-08-31_10-55-08_-16abf83e
    token = f"_p{pid}_"
    for name in os.listdir(raw_root):
        if token in name and os.path.isdir(os.path.join(raw_root, name)):
            return os.path.join(raw_root, name)
    return None


for pid in PARTICIPANTS:
    print(f"\n===== Processing P{pid} =====")

    video_dir = f"{VIDEO_FRAMES_ROOT}/P{pid}"
    base_dir = f"{GSAM_OUTPUT_ROOT}/P{pid}"

    if not os.path.isdir(video_dir):
        print(f"[WARN] video_dir not found, skip P{pid}: {video_dir}")
        continue
    if not os.path.isdir(base_dir):
        print(f"[WARN] base_dir not found, skip P{pid}: {base_dir}")
        continue

    subj_raw_dir = find_subject_folder(RAW_DATA_ROOT, pid)
    if subj_raw_dir is None:
        print(f"[WARN] raw subject folder not found for P{pid} under: {RAW_DATA_ROOT}")
        continue

    gaze_csv_path = os.path.join(subj_raw_dir, "gaze.csv")
    world_ts_path = os.path.join(subj_raw_dir, "world_timestamps.csv")

    if not (os.path.exists(gaze_csv_path) and os.path.exists(world_ts_path)):
        print(f"[WARN] Missing one of gaze/world_timestamps for P{pid}: {subj_raw_dir}")
        continue

    gaze_df = pd.read_csv(gaze_csv_path)
    world_df = pd.read_csv(world_ts_path)

    # Frame index ↔ timestamp array
    world_ts = world_df["timestamp [ns]"].to_numpy()
    final_gaze_output_csv = os.path.join(base_dir, "gaze_target_all.csv")

    # Process only folders that contain "frame" (e.g., 0frame, 500frame, ...)
    sub_folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and "frame" in f
    ]

    def process_folder(folder: str):
        folder_path = os.path.join(base_dir, folder)
        json_dir = os.path.join(folder_path, "json_data")
        mask_dir = os.path.join(folder_path, "mask_data")
        overlay_images_dir = os.path.join(folder_path, "overlay_division_result")
        video_frames_dir = os.path.join(video_dir, folder)
        gaze_output_csv = os.path.join(folder_path, "gaze_target.csv")

        os.makedirs(overlay_images_dir, exist_ok=True)

        start_frame = int(folder.replace("frame", ""))
        end_frame = start_frame + 499
 
        def get_table_mask_and_slope(mask_path, json_path):
            with open(mask_path, 'r') as f:
                rle_json = json.load(f)
            mask = rle_decode(rle_json['rle'], rle_json['shape'])
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            table_coords = []

            table_classes = ["table", "brown", "brown table"]

            for obj_id, obj in data['labels'].items():
                if obj['class_name'] in table_classes:
                    table_mask = (mask == int(obj_id))
                    y_indices, x_indices = np.where(table_mask)

                    coords = list(zip(x_indices, y_indices))
                    table_coords.extend(coords)

            if len(table_coords) < 2:
                return None, 0 

            x_all, y_all = zip(*table_coords)
            slope = np.polyfit(x_all, y_all, 1)[0]

            return None, slope
        
        def is_zero_box(obj):
            return all(obj[k] == 0 for k in ["x1", "y1", "x2", "y2"])

        def get_object_centroid_v1(objs):
            objs = [obj for obj in objs if not is_zero_box(obj)]
            if not objs:
                return None, None
            centroids = [(obj['x1'] + obj['x2']) / 2 for obj in objs]
            y_positions = [(obj['y1'] + obj['y2']) / 2 for obj in objs]
            return np.mean(centroids), np.mean(y_positions)

        def get_object_centroid_v2(objs):
            objs = [obj for obj in objs if not is_zero_box(obj)]
            if not objs:
                return None, None, None, None
            obj = max(objs, key=lambda obj: obj['x1'])
            x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            centroid_x = (x1 + x2) / 2
            top_y = y1
            return centroid_x, top_y, x1, x2

        with open(gaze_output_csv, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["timestamp", "frame_index", "auto"])

            for frame_idx in range(start_frame, end_frame + 1):
                # print("frame_idx", frame_idx)
                json_path = os.path.join(json_dir, f"mask_{frame_idx:05d}.json")
                mask_path = os.path.join(mask_dir, f"mask_{frame_idx:05d}.npy")
                image_path = os.path.join(video_frames_dir, f"{frame_idx:05d}.jpg")
                overlay_path = os.path.join(overlay_images_dir, f"{frame_idx:05d}.png")

                if not os.path.exists(json_path) or not os.path.exists(mask_path) or not os.path.exists(image_path):
                    continue

                with open(json_path, 'r') as f:
                    data = json.load(f)

                plate_objs = [obj for obj in data['labels'].values() if obj['class_name'] == 'plate' and not is_zero_box(obj)]
                lady_objs = [obj for obj in data['labels'].values() if obj['class_name'] == 'lady' and not is_zero_box(obj)]
                screen_objs = [obj for obj in data['labels'].values() if obj['class_name'] == 'screen' and not is_zero_box(obj)]

                # Step 1: remove plate inside screen
                filtered_plate_objs = []
                for plate in plate_objs:
                    px, py = (plate['x1'] + plate['x2']) / 2, (plate['y1'] + plate['y2']) / 2
                    in_screen = any(s['x1'] <= px <= s['x2'] and s['y1'] <= py <= s['y2'] for s in screen_objs)
                    if not in_screen:
                        filtered_plate_objs.append(plate)
                plate_objs = filtered_plate_objs

                # Step 2: remove lady under plate
                filtered_lady_objs = []
                for lady in lady_objs:
                    ly = (lady['y1'] + lady['y2']) / 2
                    is_under_plate = any(ly > (p['y1'] + p['y2']) / 2 for p in plate_objs)
                    if not is_under_plate:
                        filtered_lady_objs.append(lady)
                lady_objs = filtered_lady_objs

                table_mask, table_slope = get_table_mask_and_slope(mask_path, json_path)
                plate_x, plate_y = get_object_centroid_v1(plate_objs)
                lady_x, lady_y, lady_x1, lady_x2 = get_object_centroid_v2(lady_objs)
                screen_x, screen_y, _, _ = get_object_centroid_v2(screen_objs)

                img = cv2.imread(image_path)
                img_w = img.shape[1]

                if plate_x is not None and plate_y is not None:
                    intercept_y_plate = plate_y - table_slope * plate_x
                    start_y = int(table_slope * 0 + intercept_y_plate)
                    end_y = int(table_slope * img.shape[1] + intercept_y_plate)
                    cv2.line(img, (0, start_y), (img.shape[1], end_y), (255, 0, 0), 2)
                else:
                    intercept_y_plate = None

                intercept_y_div = None

                if lady_x:
                    intercept_y_div = lady_y - table_slope * lady_x
                    start_y_div = int(table_slope * 0 + intercept_y_div)
                    end_y_div = int(table_slope * img.shape[1] + intercept_y_div)
                    cv2.line(img, (0, start_y_div), (img.shape[1], end_y_div), (0, 255, 0), 2)

                if frame_idx >= len(world_ts):
                    continue

                target_ns = world_ts[frame_idx]
                gaze_row = find_nearest_gaze(gaze_df, target_ns)

                gaze_target = ""

                if gaze_row is not None:
                    gaze_x = int(gaze_row["gaze x [px]"])
                    gaze_y = int(gaze_row["gaze y [px]"])
                    cv2.circle(img, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
                    
                    y_at_gaze_plate = None
                    if intercept_y_plate is not None:
                        y_at_gaze_plate = int(table_slope * gaze_x + intercept_y_plate)
                    else:
                        y_at_gaze_plate = None

                    y_at_gaze_div = None
                    if intercept_y_div is not None:
                        y_at_gaze_div = int(table_slope * gaze_x + intercept_y_div)

                    if plate_x is not None and plate_y is not None and y_at_gaze_plate is not None and gaze_y > y_at_gaze_plate:
                        gaze_target = "near"
                    elif (
                        lady_x1 is not None and lady_x2 is not None and
                        lady_x1 <= gaze_x <= lady_x2 and
                        ((y_at_gaze_plate is None and y_at_gaze_div is not None and gaze_y >= y_at_gaze_div) or
                        (y_at_gaze_plate is not None and y_at_gaze_div is not None and y_at_gaze_plate >= gaze_y >= y_at_gaze_div))
                    ):
                        gaze_target = "middle"
                    elif ((lady_x1 is not None and lady_x2 is not None and screen_x is None 
                        and gaze_y is not None and y_at_gaze_div is not None
                        and gaze_y < y_at_gaze_div) or
                        (y_at_gaze_div is not None and gaze_y < y_at_gaze_div)
                        ): 
                        gaze_target = "far"
                    
                    csv_writer.writerow([world_ts[frame_idx], frame_idx, gaze_target])
                
                # If you want to save and inspect the overlay result images, uncomment the line below.
                # cv2.imwrite(overlay_path, img) 

        print(folder, "done")
        torch.cuda.empty_cache()
        gc.collect()

    Parallel(n_jobs=-1)(delayed(process_folder)(folder) for folder in sub_folders)

    # Merge per-folder CSVs into one file sorted by frame_idx
    csv_files = [os.path.join(base_dir, folder, "gaze_target.csv") for folder in sub_folders]
    all_data = pd.concat([pd.read_csv(f) for f in csv_files if os.path.exists(f)], ignore_index=True)

    all_data = all_data.sort_values(by="frame_index").reset_index(drop=True)

    all_data.to_csv(final_gaze_output_csv, index=False)
    print(f"CSV merge completed: {final_gaze_output_csv}")