import argparse
import pickle
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected MS-ASL clips from info.pkl archive.")
    parser.add_argument(
        "--info-tar",
        type=str,
        default="data/msasl/msasl.tar",
        help="Path to MS-ASL info tar file that contains info/info.pkl",
    )
    parser.add_argument(
        "--word-map",
        type=str,
        default="hello:hello,yes:yes,no:no,thanks:thank_you",
        help="Comma-separated mapping source_word:target_label",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save clips as data/raw/<label>/*.mp4",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/_downloads/msasl_youtube",
        help="Cache directory for full downloaded YouTube videos.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=120,
        help="Maximum number of clips to download per target label.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=12,
        help="Minimum number of frames required after trimming.",
    )
    parser.add_argument(
        "--max-variants-per-item",
        type=int,
        default=1,
        help="How many temporal variants to create from one labeled segment (>=1).",
    )
    parser.add_argument(
        "--jitter-seconds",
        type=float,
        default=0.35,
        help="Temporal jitter (seconds) used for additional variants.",
    )
    return parser.parse_args()


def parse_word_map(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid word-map entry: {chunk}. Expected source:target.")
        src, dst = chunk.split(":", 1)
        src = src.strip().lower()
        dst = dst.strip().lower()
        if not src or not dst:
            raise ValueError(f"Invalid word-map entry: {chunk}")
        out[src] = dst
    if not out:
        raise ValueError("word-map cannot be empty.")
    return out


def load_info(info_tar: Path) -> Dict:
    with tarfile.open(info_tar, "r") as tf:
        blob = tf.extractfile("info/info.pkl")
        if blob is None:
            raise ValueError("Could not find info/info.pkl in tar.")
        return pickle.loads(blob.read())


def parse_name(name: str) -> Tuple[str, float, float]:
    stem = Path(name).stem
    # Expected format: <ytid>_<start_sec_int>_<end_sec_int>
    # Use rsplit because ytid may include underscores.
    ytid, start_token, end_token = stem.rsplit("_", 2)
    start_sec = float(int(start_token))
    end_sec = float(int(end_token))
    if end_sec <= start_sec:
        end_sec = start_sec + 1.0
    return ytid, start_sec, end_sec


def find_cached_video(cache_dir: Path, ytid: str) -> Optional[Path]:
    matches = list(cache_dir.glob(f"{ytid}.*"))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def download_youtube(ytid: str, cache_dir: Path) -> Optional[Path]:
    existing = find_cached_video(cache_dir, ytid)
    if existing:
        return existing

    url = f"https://www.youtube.com/watch?v={ytid}"
    cmd = [
        "yt-dlp",
        url,
        "--no-playlist",
        "-f",
        "18/22/best",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--socket-timeout",
        "20",
        "-o",
        str(cache_dir / "%(id)s.%(ext)s"),
    ]
    rv = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if rv.returncode != 0:
        return None
    return find_cached_video(cache_dir, ytid)


def trim_to_clip(src_video: Path, dst_video: Path, start_sec: float, end_sec: float, min_frames: int) -> bool:
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    if end_frame <= start_frame:
        end_frame = start_frame + max(1, int(round(0.8 * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return False

    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(dst_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return False

    idx = start_frame
    written = 0
    while ok and idx <= end_frame:
        writer.write(frame)
        written += 1
        ok, frame = cap.read()
        idx += 1

    cap.release()
    writer.release()
    return written >= min_frames


def has_enough_frames(video_path: Path, min_frames: int) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total >= min_frames


def build_variant_windows(
    start_sec: float,
    end_sec: float,
    max_variants: int,
    jitter_seconds: float,
) -> List[Tuple[float, float]]:
    duration = max(0.8, end_sec - start_sec)
    out: List[Tuple[float, float]] = []
    needed = max(1, max_variants)
    shifts: List[float] = [0.0]
    if jitter_seconds > 0.0:
        step_idx = 1
        while len(shifts) < needed * 2:
            shifts.append(-step_idx * jitter_seconds)
            shifts.append(step_idx * jitter_seconds)
            step_idx += 1
    scales: List[float] = [1.00, 0.95, 1.05, 0.90, 1.10]

    seen = set()
    for shift in shifts:
        for scale in scales:
            s = max(0.0, start_sec + shift)
            d = max(0.8, duration * scale)
            e = s + d
            key = (round(s, 3), round(e, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append((s, e))
            if len(out) >= needed:
                return out
    return out


def main() -> None:
    args = parse_args()
    info_tar = Path(args.info_tar).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    ensure_dir(str(output_dir))
    ensure_dir(str(cache_dir))

    word_map = parse_word_map(args.word_map)
    info = load_info(info_tar)
    videos = info["videos"]

    names: List[str] = videos["name"]
    words: List[str] = videos["word"]
    if len(names) != len(words):
        raise ValueError("Corrupted info.pkl structure: name/word lengths mismatch.")

    by_label: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for name, word in zip(names, words):
        source_word = str(word).strip().lower()
        if source_word not in word_map:
            continue
        target_label = word_map[source_word]
        by_label[target_label].append((name, source_word))

    for label in sorted(set(word_map.values())):
        ensure_dir(str(output_dir / label))

    print(f"Info tar: {info_tar}")
    print(f"Output dir: {output_dir}")
    print(f"Word map: {word_map}")
    print(f"Max per label: {args.max_per_label}")
    print(f"Max variants per item: {max(1, args.max_variants_per_item)}")

    for label, items in sorted(by_label.items()):
        saved = 0
        attempted = 0
        label_dir = output_dir / label
        print(f"\n[{label}] candidates: {len(items)}")
        for name, source_word in items:
            if saved >= args.max_per_label:
                break
            attempted += 1
            try:
                ytid, start_sec, end_sec = parse_name(name)
            except Exception:
                continue

            src_video = download_youtube(ytid, cache_dir)
            if not src_video:
                continue

            windows = build_variant_windows(
                start_sec=start_sec,
                end_sec=end_sec,
                max_variants=max(1, args.max_variants_per_item),
                jitter_seconds=max(0.0, args.jitter_seconds),
            )
            for variant_idx, (v_start, v_end) in enumerate(windows):
                if saved >= args.max_per_label:
                    break

                if variant_idx == 0:
                    dst_name = f"msasl_{ytid}_{int(start_sec):06d}_{int(end_sec):06d}.mp4"
                else:
                    dst_name = (
                        f"msasl_{ytid}_{int(start_sec):06d}_{int(end_sec):06d}"
                        f"_v{variant_idx:02d}.mp4"
                    )
                dst_path = label_dir / dst_name
                if dst_path.exists():
                    if has_enough_frames(dst_path, args.min_frames):
                        saved += 1
                        continue
                    dst_path.unlink(missing_ok=True)

                ok = trim_to_clip(src_video, dst_path, v_start, v_end, min_frames=args.min_frames)
                if not ok:
                    dst_path.unlink(missing_ok=True)
                    continue

                saved += 1
                if saved % 20 == 0:
                    print(f" - saved {saved}/{args.max_per_label} ({label})")

        print(f"[{label}] saved {saved} clips (attempted {attempted})")

    print("\nMS-ASL subset download complete.")


if __name__ == "__main__":
    main()
