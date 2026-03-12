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

## Outdoor Annotation and Blurring

### 1. `ObjectDetection_GroundedDINO.py`
Runs GroundingDINO on each scene video and saves frame-level object detections (boxes + phrases + scores) into JSONL files. optionally also saves an annotated video for visual checking.

### Usage
```bash
python3 ObjectDetection_GroundedDINO.py \
    --root-dir /home/Variable-distance_viewing/Variable-distance_outdoor/Timeseries_Data_and_Scene_Video \
    --output-dir /home/Outdoor_DINO \
    --config groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --weights weights/groundingdino_swint_ogc.pth \
    --text "car . license plate" \
    --box-th 0.35 \
    --text-th 0.25 \
    --device cuda:0 \
    --every-n 1 \
    --save-annotated-video
```

### 2. `Phone_Blur.py`
Applies mosaic/pixelation to detected objects (e.g., cell phone, face) by reading the DINO JSONL detections and blurring matching boxes in the corresponding video frames. 

### Usage
```bash
python3 Phone_Blur.py \
    --video-root /home/Variable-distance_viewing/Variable-distance_outdoor/Variable-distance_outdoor_Task_Face_Mapper \
    --jsonl-root /home/od_jsonl \
    --out-root /home/mosaic_videos \
    --target-keywords "cell phone" face \
    --pixel-size 18
```

### 3. `Licenseplate_Blur.py`
Applies mosaic blur to vehicle license plates in videos using DINO JSONL detections, with an option to blur plates only when they lie inside a detected car box (to reduce false positives). 

### Usage
```bash
python3 Licenseplate_Blur.py \
    --video-root /home/original_videos \
    --jsonl-root /home/od_jsonl \
    --output-root /home/mosaic_videos \
    --pixel-block 14 \
    --plate-margin 4 \
    --extra-frames-plate 5 \
    --only-plate-inside-car
```

### 4. `Variable-Distance_Outdoor_GazeObjectMatch.py`
Assigns an object label to each gaze point by (1) aligning gaze timestamps to frame timestamps and (2) selecting the closest detection box within a radius from the DINO JSONL detections, outputs a labeled CSV per video.

### Usage
```bash
python3 Variable-Distance_Outdoor_GazeObjectMatch.py \
    --jsonl-root /home/od_jsonl \
    --gaze-root /home/Variable-distance_viewing/Variable-distance_outdoor/Timeseries_Data_and_Scene_Video \
    --out-root /home/gaze_object_labels_all \
    --radius-px 10 \
    --max-dt-ms 25
```
