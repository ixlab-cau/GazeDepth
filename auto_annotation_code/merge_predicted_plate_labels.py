import os
import pandas as pd
import argparse

# Default root
DEFAULT_PREDICT_PLATE_ROOT = "./predict_plate"      # e.g., ./predict_plate/{pid}/predicted_plate_labels.csv
DEFAULT_GSAM_OUTPUT_ROOT   = "./indoor_gaze_outputs" # e.g., ./indoor_gaze_outputs/P{pid}/gaze_target_all.csv
DEFAULT_OUTPUT_DIR         = "./final_results"      # e.g., ./final_results/P{pid}_indoor_labels.csv

PARTICIPANTS = range(1, 20)

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--predict_plate_root", type=str, default=DEFAULT_PREDICT_PLATE_ROOT,
                    help="Root folder containing predicted_plate_labels.csv (expects predict_plate_P{pid}/...)")
parser.add_argument("--gsam_output_root", type=str, default=DEFAULT_GSAM_OUTPUT_ROOT,
                    help="Root folder containing gaze_target_all.csv (expects P{pid}/...)")
parser.add_argument("--auto_label_output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                    help="Output folder to save merged CSVs")
args = parser.parse_args()

PREDICT_PLATE_ROOT = args.predict_plate_root
GSAM_OUTPUT_ROOT   = args.gsam_output_root
AUTO_LABEL_OUTPUT_DIR         = args.auto_label_output_dir

os.makedirs(AUTO_LABEL_OUTPUT_DIR, exist_ok=True)

for pid in PARTICIPANTS:
    pid_str = f"P{pid}"

    predicted_csv = f"{PREDICT_PLATE_ROOT}/{pid_str}/predicted_plate_labels.csv"
    original_csv = f"{GSAM_OUTPUT_ROOT}/{pid_str}/gaze_target_all.csv"
    output_csv = os.path.join(AUTO_LABEL_OUTPUT_DIR, f"{pid_str}_indoor_labels.csv")

    print(f"\n▶ Processing {pid_str}")

    if not os.path.exists(predicted_csv):
        print(f"❌ Predicted CSV not found: {predicted_csv}")
        continue
    if not os.path.exists(original_csv):
        print(f"❌ Original CSV not found: {original_csv}")
        continue

    df_original = pd.read_csv(original_csv)     # columns: timestamp, frame_index, auto
    df_predicted = pd.read_csv(predicted_csv)   # columns: frame_index, auto

    # Ensure types
    df_original["frame_index"] = df_original["frame_index"].astype(int)
    df_predicted["frame_index"] = df_predicted["frame_index"].astype(int)

    # Update auto by frame_index
    df_original = df_original.set_index("frame_index")
    df_predicted = df_predicted.set_index("frame_index")

    df_original.update(df_predicted)  # updates 'auto' for matching frame_index

    # Restore and enforce column order: timestamp, frame_index, auto
    df_out = df_original.reset_index()[["timestamp", "frame_index", "auto"]]
    df_out.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")