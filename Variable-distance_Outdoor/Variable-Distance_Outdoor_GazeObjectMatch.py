from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

BBoxF = Tuple[float, float, float, float]


@dataclass
class Config:
    jsonl_root: Path
    gaze_root: Path
    out_root: Path

    radius_px: int = 10
    max_dt_ms: float = 25.0
    filter_worn: bool = False
    max_jsonl_per_session: int = -1

    overwrite: bool = True
    verbose: bool = False



def session_key_from_name(folder_name: str) -> Optional[str]:
    m = re.search(r"(p\d+_.+)", folder_name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def build_session_map(root: Path) -> Dict[str, Path]:
    """
    Build mapping: session_key -> session_folder_path
    """
    mp: Dict[str, Path] = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        key = session_key_from_name(p.name)
        if key:
            mp[key] = p
    return mp


def _find_column_by_candidates(columns: Sequence[str], candidates_lower: Sequence[str]) -> Optional[str]:
    cand_set = {c.lower() for c in candidates_lower}
    for c in columns:
        if c.strip().lower() in cand_set:
            return c
    return None


def load_gaze_csv(gaze_csv: Path, filter_worn: bool = False) -> pd.DataFrame:
    df = pd.read_csv(gaze_csv)

    ts_col = _find_column_by_candidates(df.columns, ["timestamp [ns]", "timestamp_ns", "timestamp"])
    x_col = _find_column_by_candidates(df.columns, ["gaze x [px]", "gaze_x [px]", "gaze_x", "gaze x", "x"])
    y_col = _find_column_by_candidates(df.columns, ["gaze y [px]", "gaze_y [px]", "gaze_y", "gaze y", "y"])

    if ts_col is None or x_col is None or y_col is None:
        raise RuntimeError(f"Failed to find gaze.csv columns: {gaze_csv}\ncolumns={list(df.columns)}")

    out = pd.DataFrame(
        {
            "gaze_ts_ns": pd.to_numeric(df[ts_col], errors="coerce"),
            "gaze_x_px": pd.to_numeric(df[x_col], errors="coerce"),
            "gaze_y_px": pd.to_numeric(df[y_col], errors="coerce"),
        }
    )

    if filter_worn:
        worn_col = _find_column_by_candidates(df.columns, ["worn"])
        if worn_col is not None:
            worn = df[worn_col].astype(str).str.strip()
            out["worn"] = worn
            out = out[out["worn"].isin(["1", "True", "true", "TRUE"])].copy()

    out = out.dropna(subset=["gaze_ts_ns", "gaze_x_px", "gaze_y_px"]).copy()
    out["gaze_ts_ns"] = out["gaze_ts_ns"].astype(np.int64)
    out = out.sort_values("gaze_ts_ns").reset_index(drop=True)
    return out


def load_jsonl_frames(jsonl_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "frame_ts_ns" not in df.columns:
        raise RuntimeError(f"JSONL missing 'frame_ts_ns': {jsonl_path}")

    if "frame_idx" not in df.columns:
        df["frame_idx"] = np.arange(len(df), dtype=np.int64)

    df["frame_ts_ns"] = pd.to_numeric(df["frame_ts_ns"], errors="coerce")
    df = df.dropna(subset=["frame_ts_ns"]).copy()
    df["frame_ts_ns"] = df["frame_ts_ns"].astype(np.int64)

    df = df.sort_values("frame_ts_ns").reset_index(drop=True)
    return df


def clamp_box_xyxy(bb: Sequence[Any], w: int, h: int) -> Optional[BBoxF]:
    if bb is None or len(bb) != 4:
        return None

    x1, y1, x2, y2 = map(float, bb)
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def point_to_rect_dist(gx: float, gy: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = 0.0
    if gx < x1:
        dx = x1 - gx
    elif gx > x2:
        dx = gx - x2

    dy = 0.0
    if gy < y1:
        dy = y1 - gy
    elif gy > y2:
        dy = gy - y2

    return float(np.hypot(dx, dy))


def get_detection_label_text(det: Dict[str, Any]) -> str:
    for k in ("phrase", "label", "text", "class_name", "name", "caption"):
        if k in det and det[k] is not None:
            return str(det[k])
    return ""


def get_detection_bbox_xyxy(det: Dict[str, Any]) -> Optional[Sequence[Any]]:
    for k in ("bbox_xyxy_px", "bbox", "xyxy"):
        if k in det and det[k] is not None:
            bb = det[k]
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                return bb
    return None


def pick_label_by_closest(
    detections: Any,
    gaze_x: float,
    gaze_y: float,
    img_w: int,
    img_h: int,
    radius_px: int,
) -> Tuple[str, Optional[float], Optional[List[float]], Optional[str]]:
    if not isinstance(detections, list):
        return ("none", None, None, None)

    gx = float(gaze_x)
    gy = float(gaze_y)

    candidates: List[Tuple[float, float, str, List[float]]] = []

    for d in detections:
        if not isinstance(d, dict):
            continue

        phrase = get_detection_label_text(d)
        bb = get_detection_bbox_xyxy(d)
        if bb is None:
            continue

        bb2 = clamp_box_xyxy(bb, img_w, img_h)
        if bb2 is None:
            continue

        x1, y1, x2, y2 = bb2
        dist = point_to_rect_dist(gx, gy, x1, y1, x2, y2)
        if dist > radius_px:
            continue

        area = max(1.0, (x2 - x1) * (y2 - y1))
        candidates.append((dist, area, phrase, [x1, y1, x2, y2]))

    if not candidates:
        return ("none", None, None, None)

    candidates.sort(key=lambda t: (t[0], t[1]))
    best = candidates[0]
    cand_str = ";".join([f"{c[2]}@{c[0]:.1f}px" for c in candidates[:10]])

    return (best[2], float(best[0]), best[3], cand_str)

def label_one_jsonl(
    jsonl_path: Path,
    gaze_df: pd.DataFrame,
    out_csv: Path,
    radius_px: int,
    max_dt_ms: float,
) -> None:
    frames = load_jsonl_frames(jsonl_path)
    if frames.empty:
        return

    frames = frames.rename(columns={"frame_ts_ns": "ts_ns"}).copy()
    frames["ts_ns"] = frames["ts_ns"].astype(np.int64)

    merged = pd.merge_asof(
        frames.sort_values("ts_ns"),
        gaze_df.sort_values("gaze_ts_ns"),
        left_on="ts_ns",
        right_on="gaze_ts_ns",
        direction="nearest",
        tolerance=int(max_dt_ms * 1e6),  # ms -> ns
    )

    if "img_w" not in merged.columns or "img_h" not in merged.columns:
        raise RuntimeError(f"JSONL missing img_w/img_h: {jsonl_path}")

    merged["img_w"] = pd.to_numeric(merged["img_w"], errors="coerce")
    merged["img_h"] = pd.to_numeric(merged["img_h"], errors="coerce")

    out_rows: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        fi = int(row["frame_idx"])
        ts = int(row["ts_ns"])
        gx = row.get("gaze_x_px", np.nan)
        gy = row.get("gaze_y_px", np.nan)

        if pd.isna(gx) or pd.isna(gy):
            out_rows.append(
                {
                    "frame_idx": fi,
                    "frame_ts_ns": ts,
                    "gaze_x_px": None,
                    "gaze_y_px": None,
                    "label": "no_gaze_match",
                    "dist_px": None,
                    "bbox_xyxy_px": None,
                    "candidates": None,
                }
            )
            continue

        if pd.isna(row["img_w"]) or pd.isna(row["img_h"]):
            out_rows.append(
                {
                    "frame_idx": fi,
                    "frame_ts_ns": ts,
                    "gaze_x_px": float(gx),
                    "gaze_y_px": float(gy),
                    "label": "invalid_img_size",
                    "dist_px": None,
                    "bbox_xyxy_px": None,
                    "candidates": None,
                }
            )
            continue

        img_w = int(row["img_w"])
        img_h = int(row["img_h"])
        dets = row.get("detections", None)

        label, dist_px, bb, cand_str = pick_label_by_closest(
            dets, gx, gy, img_w, img_h, radius_px=radius_px
        )

        out_rows.append(
            {
                "frame_idx": fi,
                "frame_ts_ns": ts,
                "gaze_x_px": float(gx),
                "gaze_y_px": float(gy),
                "label": label,
                "dist_px": dist_px,
                "bbox_xyxy_px": bb,
                "candidates": cand_str,
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values("frame_idx")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Assign gaze object labels by matching gaze.csv with frame-level DINO JSONL detections."
    )
    parser.add_argument("--jsonl-root", type=Path, required=True, help="Root containing JSONL session folders")
    parser.add_argument("--gaze-root", type=Path, required=True, help="Root containing gaze session folders")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root directory for labeled CSV files")
    parser.add_argument("--radius-px", type=int, default=10, help="Gaze radius in pixels")
    parser.add_argument("--max-dt-ms", type=float, default=25.0, help="Max allowed frame↔gaze timestamp difference (ms)")
    parser.add_argument("--filter-worn", action="store_true", help="Use only rows with worn==1/True if gaze.csv has 'worn' column")
    parser.add_argument("--max-jsonl-per-session", type=int, default=-1, help="Limit JSONL files per session for debugging (-1 = all)")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip if output CSV already exists")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    return Config(
        jsonl_root=args.jsonl_root,
        gaze_root=args.gaze_root,
        out_root=args.out_root,
        radius_px=max(0, int(args.radius_px)),
        max_dt_ms=max(0.0, float(args.max_dt_ms)),
        filter_worn=bool(args.filter_worn),
        max_jsonl_per_session=int(args.max_jsonl_per_session),
        overwrite=not bool(args.no_overwrite),
        verbose=bool(args.verbose),
    )


def main() -> None:
    cfg = parse_args()

    if not cfg.jsonl_root.exists():
        raise FileNotFoundError(f"jsonl_root not found: {cfg.jsonl_root}")
    if not cfg.gaze_root.exists():
        raise FileNotFoundError(f"gaze_root not found: {cfg.gaze_root}")

    cfg.out_root.mkdir(parents=True, exist_ok=True)

    json_map = build_session_map(cfg.jsonl_root)
    gaze_map = build_session_map(cfg.gaze_root)

    common_keys = sorted(set(json_map.keys()) & set(gaze_map.keys()))
    if not common_keys:
        print("[ERR] No matched sessions.")
        print("  json keys sample:", list(json_map.keys())[:5])
        print("  gaze keys sample:", list(gaze_map.keys())[:5])
        return

    print(f"[INFO] Matched sessions: {len(common_keys)}")

    for key in common_keys:
        json_sess = json_map[key]
        gaze_sess = gaze_map[key]
        gaze_csv = gaze_sess / "gaze.csv"

        if not gaze_csv.exists():
            print(f"[WARN] gaze.csv missing: {gaze_csv}")
            continue

        try:
            gaze_df = load_gaze_csv(gaze_csv, filter_worn=cfg.filter_worn)
        except Exception as e:
            print(f"[WARN] Failed to load gaze.csv: {gaze_csv} -> {e}")
            continue

        jsonls = sorted(json_sess.glob("*__dino.jsonl"))
        if not jsonls:
            print(f"[WARN] No JSONL files in: {json_sess}")
            continue

        if cfg.max_jsonl_per_session > 0:
            jsonls = jsonls[: cfg.max_jsonl_per_session]

        out_sess_dir = cfg.out_root / json_sess.name
        out_sess_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {json_sess.name} ===")
        print(f"jsonl: {json_sess}")
        print(f"gaze : {gaze_csv}")
        print(f"out  : {out_sess_dir}")
        print(f"jsonl files: {len(jsonls)} | gaze rows: {len(gaze_df)}")

        for jp in tqdm(jsonls, desc=f"session {json_sess.name}", leave=False):
            stem = jp.name.replace("__dino.jsonl", "")
            out_csv = out_sess_dir / f"{stem}__gaze_object_labels.csv"

            if out_csv.exists() and not cfg.overwrite:
                if cfg.verbose:
                    print(f"[SKIP] Exists: {out_csv}")
                continue

            try:
                label_one_jsonl(
                    jp,
                    gaze_df,
                    out_csv,
                    radius_px=cfg.radius_px,
                    max_dt_ms=cfg.max_dt_ms,
                )
            except Exception as e:
                print(f"[WARN] Error on {jp.name}: {e}")

    print("\n[DONE] All processing complete.")


if __name__ == "__main__":
    main()