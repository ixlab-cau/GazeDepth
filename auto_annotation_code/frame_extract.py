import os
import supervision as sv
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse

# Default root
DEFAULT_VIDEO_ROOT = "./GazeDepth/Variable-distance_viewing/Variable-distance_indoor/Timeseries_Data_and_Scene_Video"
DEFAULT_VIDEO_FRAMES_ROOT   = "./indoor_video_frames"

FRAME_BATCH_SIZE = 500
PARTICIPANTS = range(1, 20)

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT,
                    help="Root folder containing subject folders with mp4")
parser.add_argument("--video_frames_root", type=str, default=DEFAULT_VIDEO_FRAMES_ROOT,
                    help="Output root folder to save extracted frames")
args = parser.parse_args()

VIDEO_ROOT = args.video_root
VIDEO_FRAMES_ROOT   = args.video_frames_root
os.makedirs(VIDEO_FRAMES_ROOT, exist_ok=True)

if not os.path.isdir(VIDEO_ROOT):
    raise FileNotFoundError(f"VIDEO_ROOT not found: {VIDEO_ROOT}")

for pid in PARTICIPANTS:
    # video path: find the only mp4 under the subject folder
    subj_dir = None
    for name in os.listdir(VIDEO_ROOT):
        if f"_p{pid}_" in name:
            cand = os.path.join(VIDEO_ROOT, name)
            if os.path.isdir(cand):
                subj_dir = cand
                break

    if subj_dir is None:
        print(f"[WARN] subject folder not found for P{pid} under: {VIDEO_ROOT}")
        continue

    mp4s = [f for f in os.listdir(subj_dir) if f.lower().endswith(".mp4")]
    if len(mp4s) == 0:
        print(f"[WARN] no mp4 found in: {subj_dir}")
        continue
    if len(mp4s) > 1:
        raise RuntimeError(f"Multiple mp4 files found in: {subj_dir}\n{mp4s}")

    VIDEO_PATH = os.path.join(subj_dir, mp4s[0])

    SOURCE_VIDEO_FRAME_DIR = f"{VIDEO_FRAMES_ROOT}/P{pid}"

    for path in [SOURCE_VIDEO_FRAME_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    print(f"\n=== P{pid} ===")
    print("VIDEO_PATH:", VIDEO_PATH)
    print(video_info)

    frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

    frame_index = 0
    batch_index = 0
    current_batch_dir = Path(f"{SOURCE_VIDEO_FRAME_DIR}/{batch_index}frame")
    current_batch_dir.mkdir(parents=True, exist_ok=True)

    for frame in tqdm(frame_generator, desc=f"Saving Video Frames (P{pid})"):
        frame_path = current_batch_dir / f"{frame_index:05d}.jpg"
        Image.fromarray(frame[..., ::-1]).save(frame_path)
        frame_index += 1

        if frame_index % FRAME_BATCH_SIZE == 0:
            batch_index += FRAME_BATCH_SIZE
            current_batch_dir = Path(f"{SOURCE_VIDEO_FRAME_DIR}/{batch_index}frame")
            current_batch_dir.mkdir(parents=True, exist_ok=True)