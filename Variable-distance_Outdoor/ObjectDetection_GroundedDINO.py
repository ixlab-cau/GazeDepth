from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

def _prepend_env_path(var_name: str, path_value: str) -> None:
    if not path_value:
        return
    prev = os.environ.get(var_name, "")
    if prev:
        os.environ[var_name] = f"{path_value}:{prev}"
    else:
        os.environ[var_name] = path_value


def setup_runtime_library_paths(cuda_lib_dir: Optional[str], add_torch_lib: bool = True) -> None:
    """
    Linux shared-library path workaround.
    Useful for errors like:
      ImportError: libc10.so: cannot open shared object file
    """
    if cuda_lib_dir:
        _prepend_env_path("LD_LIBRARY_PATH", cuda_lib_dir)

    if add_torch_lib:
        try:
            import torch  # local import on purpose
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            _prepend_env_path("LD_LIBRARY_PATH", torch_lib)
        except Exception:
            pass


SUBJECT_DIR_RE = re.compile(r"(?:^|[_-])p(\d+)(?:[_-]|$)", re.IGNORECASE)


def iter_subject_dirs(root: Path) -> Iterator[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir() and SUBJECT_DIR_RE.search(p.name):
            yield p


def find_mp4s(subject_dir: Path) -> List[Path]:
    return sorted([p for p in subject_dir.glob("*.mp4") if p.is_file()])


def _to_int_ns(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except Exception:
            return None


def load_world_timestamps_csv(path: Path) -> Tuple[List[int], Dict[str, List[Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"world_timestamps.csv not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]

        ts_col = None
        for cand in ("timestamp [ns]", "timestamp_ns", "timestamp", "ts_ns"):
            if cand in fieldnames:
                ts_col = cand
                break
        if ts_col is None:
            raise RuntimeError(f"timestamp column not found in {path}. columns={fieldnames}")

        rec_col = "recording id" if "recording id" in fieldnames else None
        sec_col = "section id" if "section id" in fieldnames else None

        timestamps: List[int] = []
        rec_ids: List[Any] = []
        sec_ids: List[Any] = []

        for row in reader:
            ts = _to_int_ns(row.get(ts_col))
            if ts is None:
                continue
            timestamps.append(ts)

            if rec_col:
                rec_ids.append(row.get(rec_col))
            if sec_col:
                sec_ids.append(row.get(sec_col))

    meta: Dict[str, List[Any]] = {}
    if rec_ids:
        meta["recording_ids"] = rec_ids
    if sec_ids:
        meta["section_ids"] = sec_ids

    return timestamps, meta


def cxcywh_norm_to_xyxy_px(box: Sequence[float], w: int, h: int) -> List[float]:
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h

    x1 = max(0.0, min(float(w - 1), float(x1)))
    y1 = max(0.0, min(float(h - 1), float(y1)))
    x2 = max(0.0, min(float(w - 1), float(x2)))
    y2 = max(0.0, min(float(h - 1), float(y2)))
    return [x1, y1, x2, y2]


@dataclass
class Config:
    root_dir: Path
    output_dir: Path

    text: str
    box_th: float
    text_th: float
    device: str

    gdino_config: str
    gdino_weights: str

    max_frames: int
    every_n: int
    overwrite: bool
    save_annotated_video: bool

    cuda_lib_dir: Optional[str] = None
    add_torch_lib_path: bool = True
    verbose: bool = False


class GroundingDinoRunner:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self._transform = None
        self._torch = None
        self._cv2 = None

    def load(self) -> None:
        import cv2
        import numpy as np  # noqa: F401
        import torch
        from PIL import Image  # noqa: F401
        from groundingdino.util.inference import load_model
        import groundingdino.datasets.transforms as T

        self._torch = torch
        self._cv2 = cv2
        self._transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if self.cfg.device != "cpu" and not torch.cuda.is_available():
            print("[WARN] CUDA unavailable -> fallback to cpu")
            device = "cpu"
        else:
            device = self.cfg.device

        print("[INFO] Loading GroundingDINO...")
        model = load_model(self.cfg.gdino_config, self.cfg.gdino_weights)
        model = model.to(device)
        model.eval()
        self.model = model
        self.cfg.device = device  # keep actual device
        print(f"[OK] Model loaded on {device}")

    def frame_to_dino_image(self, frame_bgr):
        from PIL import Image
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        image, _ = self._transform(pil, None)
        return image

    def predict(self, frame_bgr, caption: str, box_threshold: float, text_threshold: float):
        from groundingdino.util.inference import predict

        image = self.frame_to_dino_image(frame_bgr)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return boxes, logits, phrases


def draw_detections(frame_bgr, dets: List[Dict[str, Any]]):
    import cv2

    vis = frame_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox_xyxy_px"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f'{d["phrase"]} {d["score"]:.2f}',
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def process_video(
    runner: GroundingDinoRunner,
    subj_name: str,
    mp4_path: Path,
    timestamps_ns: List[int],
    meta: Dict[str, List[Any]],
    subj_out_dir: Path,
    cfg: Config,
) -> None:
    import cv2

    stem = mp4_path.stem
    out_jsonl = subj_out_dir / f"{stem}__dino.jsonl"
    if out_jsonl.exists() and not cfg.overwrite:
        print(f"[SKIP] Exists: {out_jsonl}")
        return

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {mp4_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(timestamps_ns) != total_frames:
        print(f"[WARN] frame count mismatch: video={total_frames}, world_ts={len(timestamps_ns)}")
        usable = min(total_frames, len(timestamps_ns))
    else:
        usable = total_frames

    if cfg.max_frames > 0:
        usable = min(usable, cfg.max_frames)

    print(
        f"[VIDEO] {mp4_path.name} | fps={fps:.3f if fps else 0.0} | "
        f"size={vw}x{vh} | frames={total_frames} | usable={usable}"
    )

    writer = None
    if cfg.save_annotated_video:
        out_avi = subj_out_dir / f"{stem}__annotated.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_avi), fourcc, fps if fps > 0 else 30.0, (vw, vh))
        if writer.isOpened():
            print(f"[INFO] annotated video -> {out_avi}")
        else:
            print(f"[WARN] failed to open annotated writer: {out_avi}")
            writer = None

    written = 0
    inferred_frames = 0
    total_dets = 0

    try:
        with out_jsonl.open("w", encoding="utf-8") as f:
            for frame_idx in tqdm(range(usable), desc=f"{subj_name}/{mp4_path.name}", leave=False):
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                run_infer = (frame_idx % max(1, cfg.every_n) == 0)

                frame_ts_ns = timestamps_ns[frame_idx]
                rec_id = meta.get("recording_ids", [None] * usable)[frame_idx] if "recording_ids" in meta and frame_idx < len(meta["recording_ids"]) else None
                sec_id = meta.get("section_ids", [None] * usable)[frame_idx] if "section_ids" in meta and frame_idx < len(meta["section_ids"]) else None

                dets: List[Dict[str, Any]] = []

                if run_infer:
                    inferred_frames += 1
                    boxes, logits, phrases = runner.predict(
                        frame_bgr,
                        caption=cfg.text,
                        box_threshold=cfg.box_th,
                        text_threshold=cfg.text_th,
                    )

                    boxes_list = boxes.detach().cpu().tolist() if hasattr(boxes, "detach") else boxes.tolist()
                    logits_list = logits.detach().cpu().tolist() if hasattr(logits, "detach") else logits.tolist()

                    def to_score(x: Any) -> float:
                        if isinstance(x, (list, tuple)):
                            return float(x[0]) if len(x) > 0 else 0.0
                        return float(x)

                    for b, s, ph in zip(boxes_list, logits_list, phrases):
                        score = to_score(s)
                        dets.append({
                            "phrase": str(ph),
                            "score": score,
                            "box_cxcywh_norm": b,
                            "bbox_xyxy_px": cxcywh_norm_to_xyxy_px(b, vw, vh),
                        })

                    total_dets += len(dets)

                if writer is not None:
                    if run_infer and dets:
                        vis = draw_detections(frame_bgr, dets)
                    else:
                        vis = frame_bgr
                    writer.write(vis)

                record = {
                    "subject_dir": subj_name,
                    "video": mp4_path.name,
                    "frame_idx": int(frame_idx),
                    "frame_ts_ns": int(frame_ts_ns),
                    "fps": float(fps) if fps else None,
                    "img_w": int(vw),
                    "img_h": int(vh),
                    "recording_id": rec_id,
                    "section_id": sec_id,
                    "prompt": cfg.text,
                    "box_th": float(cfg.box_th),
                    "text_th": float(cfg.text_th),
                    "detections": dets,
                    "ran_inference": bool(run_infer),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        print(f"[OK] wrote JSONL: {out_jsonl} (lines={written})")
        print(f"     inferred_frames={inferred_frames}, total_detections={total_dets}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()


def process_subject(runner: GroundingDinoRunner, subj_dir: Path, cfg: Config) -> None:
    subj_name = subj_dir.name
    print(f"\n=== Subject: {subj_name} ===")

    world_csv = subj_dir / "world_timestamps.csv"
    if not world_csv.exists():
        print(f"[WARN] world_timestamps.csv not found in {subj_dir}, skip")
        return

    try:
        timestamps_ns, meta = load_world_timestamps_csv(world_csv)
    except Exception as e:
        print(f"[ERR] Failed reading world_timestamps.csv: {e}")
        return

    if not timestamps_ns:
        print(f"[WARN] No timestamps in {world_csv}, skip")
        return

    mp4s = find_mp4s(subj_dir)
    if not mp4s:
        print(f"[WARN] No mp4 found in {subj_dir}, skip")
        return

    subj_out_dir = cfg.output_dir / subj_name
    subj_out_dir.mkdir(parents=True, exist_ok=True)

    for mp4_path in mp4s:
        process_video(
            runner=runner,
            subj_name=subj_name,
            mp4_path=mp4_path,
            timestamps_ns=timestamps_ns,
            meta=meta,
            subj_out_dir=subj_out_dir,
            cfg=cfg,
        )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run GroundingDINO on videos and save frame-level JSONL detections.")
    parser.add_argument("--root-dir", type=Path, required=True, help="Root folder containing subject directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save JSONL outputs (and optional annotated videos)")

    parser.add_argument("--text", type=str, default="car . license plate", help='Prompt text (e.g., "car . license plate")')
    parser.add_argument("--box-th", type=float, default=0.35, help="Box threshold")
    parser.add_argument("--text-th", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help='Inference device ("cuda:0" or "cpu")')

    parser.add_argument("--config", dest="gdino_config", type=str, required=True, help="GroundingDINO config path")
    parser.add_argument("--weights", dest="gdino_weights", type=str, required=True, help="GroundingDINO weights path")

    parser.add_argument("--max-frames", type=int, default=-1, help="Process at most N frames per video (-1 = all)")
    parser.add_argument("--every-n", type=int, default=1, help="Run inference every N frames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL outputs")
    parser.add_argument("--save-annotated-video", action="store_true", help="Save annotated AVI video")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    parser.add_argument("--cuda-lib-dir", type=str, default="/usr/local/cuda-12.1/lib64", help="CUDA lib64 path to prepend to LD_LIBRARY_PATH (Linux)")
    parser.add_argument("--no-torch-lib-path", action="store_true", help="Do not prepend torch/lib to LD_LIBRARY_PATH")

    args = parser.parse_args()

    return Config(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        text=args.text,
        box_th=float(args.box_th),
        text_th=float(args.text_th),
        device=args.device,
        gdino_config=args.gdino_config,
        gdino_weights=args.gdino_weights,
        max_frames=int(args.max_frames),
        every_n=max(1, int(args.every_n)),
        overwrite=bool(args.overwrite),
        save_annotated_video=bool(args.save_annotated_video),
        cuda_lib_dir=args.cuda_lib_dir,
        add_torch_lib_path=not bool(args.no_torch_lib_path),
        verbose=bool(args.verbose),
    )


def main() -> None:
    cfg = parse_args()

    if not cfg.root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {cfg.root_dir}")
    if not Path(cfg.gdino_weights).exists():
        raise FileNotFoundError(f"weights not found: {cfg.gdino_weights}")
    if not Path(cfg.gdino_config).exists():
        raise FileNotFoundError(f"config not found: {cfg.gdino_config}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    setup_runtime_library_paths(cfg.cuda_lib_dir, add_torch_lib=cfg.add_torch_lib_path)

    runner = GroundingDinoRunner(cfg)
    runner.load()

    subject_dirs = list(iter_subject_dirs(cfg.root_dir))
    if not subject_dirs:
        print("[ERR] Subject folders not found (expected names containing p1, p2, ...).")
        return

    print(f"[INFO] Found {len(subject_dirs)} subject dirs")

    for subj_dir in subject_dirs:
        process_subject(runner, subj_dir, cfg)

    print("\n[DONE] All processing complete.")


if __name__ == "__main__":
    main()