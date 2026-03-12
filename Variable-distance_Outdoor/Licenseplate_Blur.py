from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
Box = List[float]


@dataclass
class Config:
    video_root: Path
    jsonl_root: Path
    output_root: Path

    pixel_block: int = 14
    plate_margin: int = 4
    only_plate_inside_car: bool = True
    extra_frames_plate: int = 5
    verbose: bool = False


def mosaic_region(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, pixel_block: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(1, min(w, int(x2)))
    y2 = max(1, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    rh, rw = roi.shape[:2]
    small_w = max(1, rw // max(1, pixel_block))
    small_h = max(1, rh // max(1, pixel_block))

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic
    return img


def center_inside(box_small: Sequence[float], box_big: Sequence[float]) -> bool:
    sx1, sy1, sx2, sy2 = box_small
    bx1, by1, bx2, by2 = box_big
    cx = (sx1 + sx2) / 2.0
    cy = (sy1 + sy2) / 2.0
    return (bx1 <= cx <= bx2) and (by1 <= cy <= by2)


def clip_xyxy(box: Sequence[float], img_w: int, img_h: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(img_w - 1, float(x1)))
    y1 = max(0, min(img_h - 1, float(y1)))
    x2 = max(0, min(img_w, float(x2)))
    y2 = max(0, min(img_h, float(y2)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _convert_maybe_normalized_xyxy(vals: Sequence[float], img_w: int, img_h: int) -> Box:
    x1, y1, x2, y2 = vals
    if all(0.0 <= v <= 1.0 for v in vals):
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h
    return clip_xyxy([x1, y1, x2, y2], img_w, img_h)


def _convert_maybe_normalized_cxcywh(vals: Sequence[float], img_w: int, img_h: int) -> Box:
    cx, cy, bw, bh = vals
    if all(0.0 <= v <= 1.0 for v in vals):
        cx *= img_w
        bw *= img_w
        cy *= img_h
        bh *= img_h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return clip_xyxy([x1, y1, x2, y2], img_w, img_h)


def normalize_box_to_xyxy(box_like: Any, img_w: int, img_h: int) -> Optional[Box]:
    if box_like is None:
        return None

    if isinstance(box_like, dict):
        lower_map = {k.lower(): k for k in box_like.keys()}

        if all(k in lower_map for k in ["x1", "y1", "x2", "y2"]):
            vals = [
                float(box_like[lower_map["x1"]]),
                float(box_like[lower_map["y1"]]),
                float(box_like[lower_map["x2"]]),
                float(box_like[lower_map["y2"]]),
            ]
            return _convert_maybe_normalized_xyxy(vals, img_w, img_h)

        if all(k in lower_map for k in ["cx", "cy", "w", "h"]):
            vals = [
                float(box_like[lower_map["cx"]]),
                float(box_like[lower_map["cy"]]),
                float(box_like[lower_map["w"]]),
                float(box_like[lower_map["h"]]),
            ]
            return _convert_maybe_normalized_cxcywh(vals, img_w, img_h)

        for cand in ["bbox", "box", "xyxy", "rect"]:
            if cand in lower_map:
                return normalize_box_to_xyxy(box_like[lower_map[cand]], img_w, img_h)

        return None

    if isinstance(box_like, (list, tuple, np.ndarray)) and len(box_like) == 4:
        vals = [float(v) for v in box_like]

        if all(0.0 <= v <= 1.0 for v in vals):
            return _convert_maybe_normalized_xyxy(vals, img_w, img_h)

        x1, y1, x2, y2 = vals
        if x2 >= x1 and y2 >= y1:
            return clip_xyxy(vals, img_w, img_h)

        return _convert_maybe_normalized_cxcywh(vals, img_w, img_h)

    return None


def split_phrase_tokens(label: str) -> List[str]:
    if not label:
        return []
    s = label.lower().strip()
    for sep in ["|", ",", ";", "/", "\\"]:
        s = s.replace(sep, ".")
    tokens = [t.strip() for t in s.split(".") if t.strip()]
    return tokens if tokens else ([s] if s else [])


def get_det_labels(det: Any) -> List[str]:
    if not isinstance(det, dict):
        return []

    str_candidates = ["label", "phrase", "text", "class_name", "name", "category", "class", "caption"]
    for k in str_candidates:
        if k in det and det[k] is not None:
            raw = str(det[k]).strip().lower()
            toks = split_phrase_tokens(raw)
            return toks if toks else ([raw] if raw else [])

    list_candidates = ["labels", "classes", "phrases", "texts"]
    for k in list_candidates:
        if k in det and det[k] is not None:
            val = det[k]
            out: List[str] = []
            if isinstance(val, (list, tuple)):
                for x in val:
                    if x is None:
                        continue
                    out.extend(split_phrase_tokens(str(x).strip().lower()))
            elif isinstance(val, str):
                out.extend(split_phrase_tokens(val.strip().lower()))
            return [x for x in out if x]

    return []


def get_det_box(det: Any, img_w: int, img_h: int) -> Optional[Box]:
    if not isinstance(det, dict):
        return None

    lower_map = {k.lower(): k for k in det.keys()}

    if "bbox_xyxy_px" in lower_map:
        box = normalize_box_to_xyxy(det[lower_map["bbox_xyxy_px"]], img_w, img_h)
        if box is not None:
            return box

    if "box_cxcywh_norm" in lower_map:
        try:
            vals = [float(v) for v in det[lower_map["box_cxcywh_norm"]]]
            if len(vals) == 4:
                return _convert_maybe_normalized_cxcywh(vals, img_w, img_h)
        except Exception:
            return None

    for k in ["bbox", "box", "xyxy", "rect"]:
        if k in lower_map:
            box = normalize_box_to_xyxy(det[lower_map[k]], img_w, img_h)
            if box is not None:
                return box

    if "boxes" in lower_map:
        boxes_val = det[lower_map["boxes"]]
        if isinstance(boxes_val, (list, tuple, np.ndarray)) and len(boxes_val) == 4:
            box = normalize_box_to_xyxy(boxes_val, img_w, img_h)
            if box is not None:
                return box

    return normalize_box_to_xyxy(det, img_w, img_h)


def is_car_label(label: str) -> bool:
    label = (label or "").lower()
    return (
        "car" in label
        or "vehicle" in label
        or label in {"sedan", "suv", "van", "truck", "bus"}
    )


def is_plate_label(label: str) -> bool:
    label = (label or "").lower().strip()
    return (
        "license plate" in label
        or "licence plate" in label
        or "number plate" in label
        or label == "plate"
        or "car plate" in label
    )


def dedup_boxes(boxes: Iterable[Sequence[float]]) -> List[Box]:
    out: List[Box] = []
    seen = set()
    for b in boxes:
        key = tuple(int(round(v)) for v in b)
        if key not in seen:
            seen.add(key)
            out.append([float(v) for v in b])
    return out


def load_jsonl_by_frame(jsonl_path: Path, verbose: bool = False) -> Dict[int, Dict[str, Any]]:
    frame_map: Dict[int, Dict[str, Any]] = {}

    total_lines = 0
    nonempty_det_frames = 0
    total_cars = 0
    total_plates = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                rec = json.loads(line)
            except Exception as e:
                if verbose:
                    print(f"[WARN] JSON decode failed: {jsonl_path} line {line_no}: {e}")
                continue

            frame_idx = rec.get("frame_idx")
            if frame_idx is None:
                continue

            img_w = rec.get("img_w")
            img_h = rec.get("img_h")
            if img_w is None or img_h is None:
                continue
            img_w, img_h = int(img_w), int(img_h)

            detections = rec.get("detections", []) or []
            if detections:
                nonempty_det_frames += 1

            cars: List[Box] = []
            plates: List[Box] = []

            for det in detections:
                box = get_det_box(det, img_w, img_h)
                if box is None:
                    continue

                labels = get_det_labels(det)
                if not labels:
                    continue

                if any(is_plate_label(lbl) for lbl in labels):
                    plates.append(box)
                    continue

                if any(is_car_label(lbl) for lbl in labels):
                    cars.append(box)

            total_cars += len(cars)
            total_plates += len(plates)

            frame_map[int(frame_idx)] = {
                "cars": cars,
                "plates": plates,
                "img_w": img_w,
                "img_h": img_h,
            }

    if verbose:
        print(
            f"[JSONL] {jsonl_path.name} | lines={total_lines}, nonempty={nonempty_det_frames}, "
            f"cars={total_cars}, plates={total_plates}"
        )

    return frame_map

def process_video(video_path: Path, jsonl_path: Path, out_path: Path, cfg: Config) -> None:
    """Apply plate mosaic to one video using matching JSONL detections."""
    frame_map = load_jsonl_by_frame(jsonl_path, verbose=cfg.verbose)
    if not frame_map:
        print(f"[SKIP] Empty frame map: {jsonl_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERR] Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[ERR] Failed to open VideoWriter: {out_path}")
        cap.release()
        return

    pbar = tqdm(total=total, desc=video_path.name, leave=False)

    frame_idx = 0
    applied_count = 0
    frames_with_info = 0
    frames_with_car = 0
    frames_with_plate = 0
    plate_inside_hits = 0
    frames_with_temporal_plate = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        temporal_plates: List[Box] = []
        temporal_cars: List[Box] = []

        i0 = max(0, frame_idx - cfg.extra_frames_plate)
        i1 = min(total - 1, frame_idx + cfg.extra_frames_plate)

        for j in range(i0, i1 + 1):
            info_j = frame_map.get(j)
            if info_j is None:
                continue

            if j == frame_idx:
                frames_with_info += 1
                if info_j["cars"]:
                    frames_with_car += 1
                if info_j["plates"]:
                    frames_with_plate += 1

            temporal_cars.extend(info_j.get("cars", []))
            temporal_plates.extend(info_j.get("plates", []))

        if temporal_plates:
            frames_with_temporal_plate += 1

        uniq_plates = dedup_boxes(temporal_plates)
        uniq_cars = dedup_boxes(temporal_cars)

        for plate_box in uniq_plates:
            use_this_plate = True

            if cfg.only_plate_inside_car:
                if not uniq_cars:
                    use_this_plate = False
                else:
                    inside_any_car = any(center_inside(plate_box, car_box) for car_box in uniq_cars)
                    if inside_any_car:
                        plate_inside_hits += 1
                    else:
                        use_this_plate = False

            if not use_this_plate:
                continue

            x1, y1, x2, y2 = plate_box
            x1 -= cfg.plate_margin
            y1 -= cfg.plate_margin
            x2 += cfg.plate_margin
            y2 += cfg.plate_margin

            frame = mosaic_region(frame, x1, y1, x2, y2, pixel_block=cfg.pixel_block)
            applied_count += 1

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    print(f"[DONE] {video_path.name} -> {out_path.name}")
    print(f"  frames={frame_idx}, json_frames={len(frame_map)}, frames_with_info={frames_with_info}")
    print(f"  frames_with_car={frames_with_car}, frames_with_plate={frames_with_plate}")
    print(f"  frames_with_temporal_plate={frames_with_temporal_plate}")
    print(f"  plate_inside_hits={plate_inside_hits}, mosaics={applied_count}")


def canonical_video_stem_for_jsonl(video_file: Path) -> str:
    stem = video_file.stem

    removable_suffixes = [
        "_mosaic__phoneblur",
        "__phoneblur",
        "_phoneblur",
        "_mosaic",
        "__mosaic",
        "_faceblur",
        "__faceblur",
        "_plateblur",
        "__plateblur",
    ]

    changed = True
    while changed:
        changed = False
        for sfx in removable_suffixes:
            if stem.endswith(sfx):
                stem = stem[: -len(sfx)]
                changed = True

    return stem


def find_matching_jsonl(subject_json_dir: Path, video_file: Path, verbose: bool = False) -> Optional[Path]:
    stem_raw = video_file.stem
    stem = canonical_video_stem_for_jsonl(video_file)

    exact = subject_json_dir / f"{stem}__dino.jsonl"
    if exact.exists():
        return exact

    exact_raw = subject_json_dir / f"{stem_raw}__dino.jsonl"
    if exact_raw.exists():
        return exact_raw

    cands = list(subject_json_dir.glob(f"{stem}*__dino.jsonl"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        cands = sorted(cands, key=lambda p: (len(p.stem), p.name))
        return cands[0]

    if verbose:
        print(
            f"[DEBUG] JSONL not found for video={video_file.name} | "
            f"raw_stem={stem_raw} | normalized_stem={stem}"
        )
    return None


def make_output_path(out_dir: Path, video_file: Path) -> Path:
    return out_dir / f"{video_file.stem}__plateblur{video_file.suffix}"

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Apply mosaic blur to vehicle license plates using DINO JSONL detections.")
    parser.add_argument("--video-root", type=Path, required=True, help="Root directory containing subject video folders")
    parser.add_argument("--jsonl-root", type=Path, required=True, help="Root directory containing subject JSONL folders")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--pixel-block", type=int, default=14, help="Mosaic strength (smaller = stronger)")
    parser.add_argument("--plate-margin", type=int, default=4, help="Plate bbox margin in pixels")
    parser.add_argument("--extra-frames-plate", type=int, default=5, help="Temporal buffering radius (±N frames)")
    parser.add_argument("--only-plate-inside-car", action="store_true", help="Mosaic only plates whose center lies inside a detected car box")
    parser.add_argument("--all-plates", action="store_true", help="Mosaic all detected plates (overrides --only-plate-inside-car)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    only_inside = True
    if args.all_plates:
        only_inside = False
    elif args.only_plate_inside_car:
        only_inside = True

    return Config(
        video_root=args.video_root,
        jsonl_root=args.jsonl_root,
        output_root=args.output_root,
        pixel_block=max(1, args.pixel_block),
        plate_margin=max(0, args.plate_margin),
        only_plate_inside_car=only_inside,
        extra_frames_plate=max(0, args.extra_frames_plate),
        verbose=args.verbose,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([p for p in cfg.video_root.iterdir() if p.is_dir()])
    print(f"Found subject dirs: {len(subject_dirs)}")

    for subject_video_dir in subject_dirs:
        subject_name = subject_video_dir.name
        subject_json_dir = cfg.jsonl_root / subject_name

        if not subject_json_dir.exists():
            print(f"[SKIP] Missing JSONL folder: {subject_json_dir}")
            continue

        video_files = sorted(
            [p for p in subject_video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        )
        if not video_files:
            print(f"[SKIP] No videos: {subject_video_dir}")
            continue

        for video_file in video_files:
            jsonl_path = find_matching_jsonl(subject_json_dir, video_file, verbose=cfg.verbose)
            if jsonl_path is None:
                print(f"[SKIP] No matching JSONL: {video_file}")
                continue

            out_dir = cfg.output_root / subject_name
            out_path = make_output_path(out_dir, video_file)
            process_video(video_file, jsonl_path, out_path, cfg)


if __name__ == "__main__":
    main()