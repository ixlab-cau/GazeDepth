# GazeDepth

## Indoor Automatic Annotation Process

### Run order (required)
These scripts **must** be executed in the following order.

---

### 1. `Frame_Extract.py`
Extracts video frames from each participant’s scene video and saves them under:
- `{VIDEO_FRAMES_ROOT}/P{pid}/` (e.g., `P1/0frame`, `P1/500frame`, ...)

#### Usage
```bash
python3 Frame_Extract.py \
  --video_root "/path/to/Timeseries_Data_and_Scene_Video" \
  --video_frames_root "/path/to/indoor_video_frames"
```

### 2. `ObjectDetection_Grounded-SAM2.py`
Runs **Grounded-SAM2** on each extracted frame in `{VIDEO_FRAMES_ROOT}/` and saves per-frame outputs to:
- `{GSAM_OUTPUT_ROOT}/P{pid}/...` (RLE masks, JSON metadata, and visualization/result images)

#### Usage
```bash
python3 ObjectDetection_Grounded-SAM2.py \
  --video_frames_root "/path/to/indoor_video_frames" \
  --gsam_output_root "/path/to/indoor_gaze_outputs" \
  --sam2_checkpoint "/path/to/checkpoints/sam2.1_hiera_large.pt" \
  --sam2_cfg "configs/sam2.1/sam2.1_hiera_l.yaml"
```

### 3. `Area_Division.py`
Loads detection outputs from `{GSAM_OUTPUT_ROOT}/`, applies the **area-division rules described in the Automatic Annotation section of our paper**, and writes `gaze_target_all.csv` inside each participant folder:
- `{GSAM_OUTPUT_ROOT}/P{pid}/gaze_target_all.csv`

Output columns:
- `timestamp`, `frame_index`, `auto`

#### Usage
```bash
python3 Area_Division.py \
  --video_frames_root "/path/to/indoor_video_frames" \
  --gsam_output_root "/path/to/indoor_gaze_outputs"
```

### 4. `Predict_Plate_Position.py`
Handles the **“missing plate”** exception case. For frames where **lady is detected but plate is not detected**, it predicts the plate position following the rules described in the **Automatic Annotation** section of our paper, saves overlay images under `predict_plate/` for each participant, and writes `predicted_plate_labels.csv` containing additional auto labels.

Output per participant:
- `{PREDICT_PLATE_ROOT}/P{pid}/predicted_plate_labels.csv` with columns: `frame_index`, `auto`

#### Usage
```bash
python3 Predict_Plate_Position.py \
  --video_frames_root "/path/to/indoor_video_frames" \
  --gsam_output_root "/path/to/indoor_gaze_outputs" \
  --predict_plate_output_root "/path/to/predict_plate"
```

### 5. `Merge_Predicted_Plate_Labels.py`
Merges `predicted_plate_labels.csv` with the original `gaze_target_all.csv` and saves the final per-participant labels to `final_results/` as:
- `{AUTO_LABEL_OUTPUT_DIR}/P{pid}_indoor_labels.csv`

#### Usage
```bash
python3 Merge_Predicted_Plate_Labels.py \
  --predict_plate_root "/path/to/predict_plate" \
  --gsam_output_root "/path/to/indoor_gaze_outputs" \
  --auto_label_output_dir "/path/to/final_results"
```
