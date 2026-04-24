import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate raw clips into a cleaner, consistent training set.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Input raw directory with label subfolders.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for curated clips.")
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated labels to curate. Empty means all labels in input-dir.",
    )
    parser.add_argument("--target-per-label", type=int, default=150, help="Max curated clips per label.")
    parser.add_argument("--min-duration", type=float, default=0.75, help="Minimum clip duration in seconds.")
    parser.add_argument("--max-duration", type=float, default=3.80, help="Maximum clip duration in seconds.")
    parser.add_argument("--min-frames", type=int, default=16, help="Minimum frame count.")
    parser.add_argument("--min-motion", type=float, default=0.010, help="Minimum average motion score.")
    parser.add_argument(
        "--max-variants-per-base",
        type=int,
        default=4,
        help="Max clips kept per base source item (to limit near-duplicates).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".mp4,.avi,.mov,.mkv",
        help="Comma-separated video extensions.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report without copying clips.")
    return parser.parse_args()


def parse_labels(raw: str) -> List[str]:
    labels = [x.strip() for x in raw.split(",")]
    return [x for x in labels if x]


def parse_extensions(raw: str) -> Tuple[str, ...]:
    exts = [x.strip().lower() for x in raw.split(",")]
    exts = [x for x in exts if x]
    return tuple(exts)


def iter_label_dirs(input_dir: Path, labels: Sequence[str]) -> List[Path]:
    if labels:
        out = []
        for label in labels:
            p = input_dir / label
            if p.exists() and p.is_dir():
                out.append(p)
        return out
    return [p for p in sorted(input_dir.iterdir()) if p.is_dir()]


def list_videos(label_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    return [p for p in sorted(label_dir.iterdir()) if p.is_file() and p.suffix.lower() in extensions]


def motion_score_for_clip(video_path: Path, sample_step: int = 2) -> Optional[Tuple[int, float, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0 or frames_total <= 0:
        cap.release()
        return None

    prev = None
    idx = 0
    diffs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_step != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            diffs.append(float(diff.mean()) / 255.0)
        prev = gray
        idx += 1
    cap.release()

    duration = frames_total / fps
    motion = float(sum(diffs) / len(diffs)) if diffs else 0.0
    return frames_total, duration, motion


def base_key_for_clip(path: Path) -> str:
    stem = path.stem
    # Remove variant suffix for MS-ASL jitter clips.
    stem = re.sub(r"_v\d{2}$", "", stem)
    return stem


def quality_rank(duration: float, motion: float) -> Tuple[float, float]:
    # Prefer durations close to isolated-sign range and stronger motion.
    return (abs(duration - 2.0), -motion)


def curate_label(
    videos: List[Path],
    target_per_label: int,
    min_duration: float,
    max_duration: float,
    min_frames: int,
    min_motion: float,
    max_variants_per_base: int,
) -> Tuple[List[Dict], List[Dict]]:
    kept: List[Dict] = []
    rejected: List[Dict] = []

    analyzed: List[Dict] = []
    for video in videos:
        info = motion_score_for_clip(video)
        if info is None:
            rejected.append({"path": str(video), "reason": "decode_failed"})
            continue
        frames_total, duration, motion = info
        row = {
            "path": str(video),
            "frames": frames_total,
            "duration": duration,
            "motion": motion,
            "base_key": base_key_for_clip(video),
        }

        if frames_total < min_frames:
            row["reason"] = "too_few_frames"
            rejected.append(row)
            continue
        if duration < min_duration or duration > max_duration:
            row["reason"] = "duration_out_of_range"
            rejected.append(row)
            continue
        if motion < min_motion:
            row["reason"] = "low_motion"
            rejected.append(row)
            continue
        analyzed.append(row)

    analyzed.sort(key=lambda r: quality_rank(float(r["duration"]), float(r["motion"])))

    by_base: Dict[str, int] = defaultdict(int)
    for row in analyzed:
        base_key = str(row["base_key"])
        if by_base[base_key] >= max_variants_per_base:
            row["reason"] = "variant_cap"
            rejected.append(row)
            continue
        by_base[base_key] += 1
        kept.append(row)
        if len(kept) >= target_per_label:
            break

    # Remaining valid-but-not-selected are also rejections by capacity.
    if len(kept) < len(analyzed):
        selected_paths = {r["path"] for r in kept}
        for row in analyzed:
            if row["path"] in selected_paths:
                continue
            if "reason" not in row:
                row["reason"] = "capacity_cut"
                rejected.append(row)

    return kept, rejected


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    labels = parse_labels(args.labels)
    extensions = parse_extensions(args.extensions)

    if not input_dir.exists():
        raise ValueError(f"Input dir does not exist: {input_dir}")
    if not extensions:
        raise ValueError("No valid extensions provided.")
    if args.target_per_label <= 0:
        raise ValueError("--target-per-label must be > 0")

    label_dirs = iter_label_dirs(input_dir, labels)
    if not label_dirs:
        raise ValueError("No label directories found to curate.")

    ensure_dir(str(output_dir))
    report_rows: List[Dict] = []
    reject_rows: List[Dict] = []

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Labels: {[p.name for p in label_dirs]}")
    print(f"Target per label: {args.target_per_label}")
    print(
        "Filters: "
        f"duration=[{args.min_duration:.2f}, {args.max_duration:.2f}] "
        f"min_frames={args.min_frames} min_motion={args.min_motion:.4f} "
        f"max_variants_per_base={args.max_variants_per_base}"
    )

    for label_dir in label_dirs:
        label = label_dir.name
        videos = list_videos(label_dir, extensions)
        kept, rejected = curate_label(
            videos=videos,
            target_per_label=args.target_per_label,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_frames=args.min_frames,
            min_motion=args.min_motion,
            max_variants_per_base=args.max_variants_per_base,
        )

        dst_label_dir = output_dir / label
        if not args.dry_run:
            ensure_dir(str(dst_label_dir))
            for row in kept:
                src = Path(str(row["path"]))
                dst = dst_label_dir / src.name
                shutil.copy2(src, dst)

        durations = [float(r["duration"]) for r in kept]
        motions = [float(r["motion"]) for r in kept]
        report_rows.append(
            {
                "label": label,
                "input_count": len(videos),
                "kept_count": len(kept),
                "rejected_count": len(rejected),
                "kept_min_duration": f"{min(durations):.3f}" if durations else "",
                "kept_max_duration": f"{max(durations):.3f}" if durations else "",
                "kept_avg_motion": f"{(sum(motions)/len(motions)):.5f}" if motions else "",
            }
        )
        for row in rejected:
            reject_rows.append(
                {
                    "label": label,
                    "path": row.get("path", ""),
                    "reason": row.get("reason", "unknown"),
                    "duration": row.get("duration", ""),
                    "motion": row.get("motion", ""),
                    "frames": row.get("frames", ""),
                    "base_key": row.get("base_key", ""),
                }
            )

        print(
            f"[{label}] input={len(videos)} kept={len(kept)} rejected={len(rejected)} "
            f"({'dry-run' if args.dry_run else 'copied'})"
        )

    report_path = output_dir / "curation_summary.csv"
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "input_count",
                "kept_count",
                "rejected_count",
                "kept_min_duration",
                "kept_max_duration",
                "kept_avg_motion",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    reject_path = output_dir / "curation_rejections.csv"
    with open(reject_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "path", "reason", "duration", "motion", "frames", "base_key"],
        )
        writer.writeheader()
        writer.writerows(reject_rows)

    print(f"Saved summary: {report_path}")
    print(f"Saved rejections: {reject_path}")


if __name__ == "__main__":
    main()

