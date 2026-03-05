ObjectDetection_GroundedDINO.py — Runs GroundingDINO on each scene video and saves frame-level object detections (boxes + phrases + scores) into JSONL files. optionally also saves an annotated video for visual checking.

PhoneBlur.py — Applies mosaic/pixelation to detected objects (e.g., cell phone, face) by reading the DINO JSONL detections and blurring matching boxes in the corresponding video frames. 

LicenseplateBlur.py — Applies mosaic blur to vehicle license plates in videos using DINO JSONL detections, with an option to blur plates only when they lie inside a detected car box (to reduce false positives). 

Variable-Distance_Outdoor_GazeObjectMatch.py — Assigns an object label to each gaze point by (1) aligning gaze timestamps to frame timestamps and (2) selecting the closest detection box within a radius from the DINO JSONL detections, outputs a labeled CSV per video.

The usage examples are included in each script.