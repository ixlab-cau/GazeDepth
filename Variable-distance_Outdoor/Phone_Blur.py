from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
BBox = Tuple[int, int, int, int]

@dataclass
class Config:
    video_root: Path
    jsonl_root: Path
    out_root: Path

    target_keywords: List[str]
    pixel_size: int = 18

    overwrite: bool = False
    verbose: bool = False

SESSION_KEY_RE = re.compile(r"(p\d+_.+)$", re.IGNORECASE)


def session_key_from_name(folder_name: str) -> Optional[str]:
    m = re.search(r"(p\d+_.+)", folder_name, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1)


def build_json_session_map(jsonl_root: Path) -> Dict[str, Path]:
    mp: Dict[str, Path] = {}
    for p in jsonl_root.iterdir():
        if not p.is_dir():
            continue
        key = session_key_from_name(p.name)
        if key:
            mp[key] = p
    return mp


def keyword_match(text: str, target_keywords: Sequence[str]) -> bool:
    s = (text or "").lower()
    return any(k.lower() in s for k in target_keywords)


def clamp_box_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[BBox]:

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def apply_mosaic(frame_bgr, x1: int, y1: int, x2: int, y2: int, pixel_size: int) -> None:
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return

    pixel_size = max(1, int(pixel_size))

    small_w = max(1, (x2 - x1) // pixel_size)
    small_h = max(1, (y2 - y1) // pixel_size)

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame_bgr[y1:y2, x1:x2] = mosaic


def get_detection_text(det: dict) -> str:
    for key in ("phrase", "label", "text", "class_name", "name", "caption"):
        if key in det and det[key] is not None:
            return str(det[key])
    return ""


def get_detection_bbox_xyxy(det: dict) -> Optional[BBox]:
    bb = det.get("bbox_xyxy_px")
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None

    try:
        return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    except Exception:
        return None


def iter_needed_boxes_from_jsonl(
    jsonl_path: Path,
    target_keywords: Sequence[str],
) -> Iterator[Tuple[int, List[BBox]]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                fi = int(obj.get("frame_idx", -1))
            except Exception:
                fi = -1

            if fi < 0:
                continue

            dets = obj.get("detections", []) or []
            boxes: List[BBox] = []

            if isinstance(dets, list):
                for d in dets:
                    if not isinstance(d, dict):
                        continue

                    text = get_detection_text(d)
                    if not keyword_match(text, target_keywords):
                        continue

                    bb = get_detection_bbox_xyxy(d)
                    if bb is None:
                        continue

                    boxes.append(bb)

            yield fi, boxes


def process_one_video(mp4_path: Path, jsonl_path: Path, out_path: Path, cfg: Config) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not cfg.overwrite:
        print(f"[SKIP] Output exists: {out_path}")
        return

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {mp4_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 30.0, (w, h))
    if not writer.isOpened():
        print(f"[ERR] Cannot open writer: {out_path}")
        cap.release()
        return

    json_iter = iter_needed_boxes_from_jsonl(jsonl_path, cfg.target_keywords)
    next_json = next(json_iter, None)

    frame_idx = 0
    frames_with_boxes = 0
    applied_boxes = 0

    pbar = tqdm(total=total_frames, desc=f"mosaic {mp4_path.name}", leave=False)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            boxes: List[BBox] = []

            while next_json is not None and next_json[0] < frame_idx:
                next_json = next(json_iter, None)

            if next_json is not None and next_json[0] == frame_idx:
                boxes = next_json[1]
                next_json = next(json_iter, None)

            if boxes:
                frames_with_boxes += 1

            for (x1, y1, x2, y2) in boxes:
                c = clamp_box_xyxy(x1, y1, x2, y2, w, h)
                if c is None:
                    continue
                apply_mosaic(frame, *c, pixel_size=cfg.pixel_size)
                applied_boxes += 1

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    print(f"[DONE] {mp4_path.name} -> {out_path.name}")
    print(f"  frames={frame_idx}, frames_with_boxes={frames_with_boxes}, mosaics={applied_boxes}")


def candidate_stems_from_video_stem(stem: str) -> List[str]:
    suffixes = [
        "_mosaic", "__mosaic", "-mosaic",
        "_annotated", "__annotated",
        "__phoneblur", "_phoneblur",
        "__faceblur", "_faceblur",
    ]

    cands = [stem]
    for sfx in suffixes:
        if stem.endswith(sfx):
            cands.append(stem[: -len(sfx)])

    out: List[str] = []
    seen = set()
    for s in cands:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def find_matching_jsonl_for_video(j_sess: Path, mp4: Path) -> Optional[Path]:
    stem_candidates = candidate_stems_from_video_stem(mp4.stem)

    for st in stem_candidates:
        cand = j_sess / f"{st}__dino.jsonl"
        if cand.exists():
            return cand

    return None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Apply mosaic to DINO detections matching target keywords (e.g., cell phone, face)."
    )
    parser.add_argument("--video-root", type=Path, required=True, help="Root directory containing video session folders")
    parser.add_argument("--jsonl-root", type=Path, required=True, help="Root directory containing JSONL session folders")
    parser.add_argument("--out-root", type=Path, required=True, help="Output directory root")
    parser.add_argument(
        "--target-keywords",
        type=str,
        nargs="+",
        default=["cell phone", "face"],
        help='Keyword substrings to match detection labels/phrases (e.g., --target-keywords "cell phone" face)',
    )
    parser.add_argument("--pixel-size", type=int, default=18, help="Mosaic strength (smaller = stronger)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    return Config(
        video_root=args.video_root,
        jsonl_root=args.jsonl_root,
        out_root=args.out_root,
        target_keywords=[str(k).lower() for k in args.target_keywords],
        pixel_size=max(1, int(args.pixel_size)),
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
    )


def main() -> None:
    cfg = parse_args()

    if not cfg.video_root.exists():
        raise FileNotFoundError(f"video root not found: {cfg.video_root}")
    if not cfg.jsonl_root.exists():
        raise FileNotFoundError(f"jsonl root not found: {cfg.jsonl_root}")

    cfg.out_root.mkdir(parents=True, exist_ok=True)

    json_session_map = build_json_session_map(cfg.jsonl_root)
    if not json_session_map:
        print("[ERR] No JSONL session folders found.")
        return

    video_sessions = [p for p in cfg.video_root.iterdir() if p.is_dir()]
    if not video_sessions:
        print("[ERR] No video session folders found.")
        return

    print(f"[INFO] video sessions: {len(video_sessions)}")
    print(f"[INFO] json sessions : {len(json_session_map)}")

    for v_sess in sorted(video_sessions):
        key = session_key_from_name(v_sess.name)
        if not key:
            if cfg.verbose:
                print(f"[SKIP] Cannot extract session key: {v_sess.name}")
            continue

        j_sess = json_session_map.get(key)
        if j_sess is None:
            print(f"[WARN] No JSON session match for: {v_sess.name} (key={key})")
            continue

        videos = sorted([p for p in v_sess.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        if not videos:
            print(f"[WARN] No videos in session: {v_sess}")
            continue

        print("\n=== Session matched ===")
        print(f"video: {v_sess}")
        print(f"json : {j_sess}")

        for video_path in videos:
            jsonl_path = find_matching_jsonl_for_video(j_sess, video_path)
            if jsonl_path is None:
                tried = [f"{st}__dino.jsonl" for st in candidate_stems_from_video_stem(video_path.stem)]
                print(f"[WARN] JSONL missing for {video_path.name}: tried {tried}")
                continue

            out_dir = cfg.out_root / j_sess.name
            out_video = out_dir / f"{video_path.stem}__phoneblur{video_path.suffix}"

            process_one_video(video_path, jsonl_path, out_video, cfg)

    print("\n[DONE] All processing complete.")


if __name__ == "__main__":
    main()