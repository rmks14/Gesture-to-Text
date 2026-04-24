import argparse
import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from .utils import ensure_dir


DEFAULT_GLOSSES = ["hello", "yes", "no", "thank you"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small real ASL subset from WLASL.")
    parser.add_argument(
        "--index-file",
        type=str,
        default="tmp_wlasl/start_kit/WLASL_v0.3.json",
        help="Path to WLASL_v0.3.json",
    )
    parser.add_argument(
        "--glosses",
        type=str,
        default=",".join(DEFAULT_GLOSSES),
        help="Comma-separated exact gloss names, e.g. hello,yes,no,thank you",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Target directory organized as data/raw/<label>/*.mp4",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/_downloads",
        help="Cache directory for downloaded source videos.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=12,
        help="Max clips to save per label.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=40,
        help="Network timeout in seconds for direct URL downloads.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=10,
        help="Discard clips shorter than this many frames.",
    )
    return parser.parse_args()


def load_index(index_file: Path) -> List[Dict]:
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_glosses(glosses_csv: str) -> List[str]:
    glosses = [g.strip() for g in glosses_csv.split(",")]
    glosses = [g for g in glosses if g]
    if not glosses:
        raise ValueError("No valid glosses provided.")
    return glosses


def safe_label(gloss: str) -> str:
    label = gloss.strip().lower().replace(" ", "_").replace("/", "_")
    return "".join(ch for ch in label if ch.isalnum() or ch in {"_", "-"})


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def youtube_id(url: str) -> Optional[str]:
    if "youtu.be/" in url:
        return url.rsplit("/", 1)[-1][:11]
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    values = query.get("v", [])
    if values:
        return values[0][:11]
    return None


def download_direct(url: str, dst_path: Path, timeout: int) -> bool:
    if dst_path.exists():
        return True
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(dst_path, "wb") as f:
            f.write(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def find_downloaded_youtube_file(cache_dir: Path, yt_id: str) -> Optional[Path]:
    matches = list(cache_dir.glob(f"{yt_id}.*"))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def download_youtube(url: str, cache_dir: Path) -> Optional[Path]:
    yt_id = youtube_id(url)
    if not yt_id:
        return None
    existing = find_downloaded_youtube_file(cache_dir, yt_id)
    if existing and existing.is_file():
        return existing

    cmd = [
        "yt-dlp",
        url,
        "--no-playlist",
        "-f",
        "18/22/best",
        "-o",
        str(cache_dir / "%(id)s.%(ext)s"),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return find_downloaded_youtube_file(cache_dir, yt_id)


def copy_or_trim(
    src_video: Path,
    dst_video: Path,
    frame_start: int,
    frame_end: int,
    min_frames: int,
) -> bool:
    if frame_end <= 0:
        shutil.copyfile(src_video, dst_video)
        return count_frames(dst_video) >= min_frames

    start = max(0, frame_start - 1)
    end = max(start, frame_end - 1)

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    ok, frame = cap.read()
    if not ok:
        cap.release()
        return False

    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(dst_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return False

    idx = start
    written = 0
    while ok and idx <= end:
        writer.write(frame)
        written += 1
        ok, frame = cap.read()
        idx += 1

    cap.release()
    writer.release()
    return written >= min_frames


def count_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, total)


def pick_entries(content: List[Dict], glosses: Iterable[str]) -> Dict[str, List[Dict]]:
    by_gloss: Dict[str, List[Dict]] = {}
    for gloss in glosses:
        entry = next((e for e in content if e.get("gloss") == gloss), None)
        by_gloss[gloss] = entry.get("instances", []) if entry else []
    return by_gloss


def main() -> None:
    args = parse_args()
    index_file = Path(args.index_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    direct_cache = cache_dir / "direct"
    youtube_cache = cache_dir / "youtube"
    ensure_dir(str(output_dir))
    ensure_dir(str(direct_cache))
    ensure_dir(str(youtube_cache))

    glosses = parse_glosses(args.glosses)
    content = load_index(index_file)
    selected = pick_entries(content, glosses)

    print(f"Index: {index_file}")
    print(f"Output: {output_dir}")
    print(f"Glosses: {glosses}")

    for gloss in glosses:
        instances = selected.get(gloss, [])
        label = safe_label(gloss)
        label_dir = output_dir / label
        ensure_dir(str(label_dir))

        saved = 0
        attempts = 0
        print(f"\n[{gloss}] instances available: {len(instances)}")

        for inst in instances:
            if saved >= args.max_per_label:
                break
            url = inst["url"]
            vid = inst["video_id"]
            frame_start = int(inst.get("frame_start", 1))
            frame_end = int(inst.get("frame_end", -1))
            attempts += 1

            if url.lower().endswith(".swf"):
                print(f" - skip {vid}: swf")
                continue

            src_path: Optional[Path] = None
            if is_youtube(url):
                src_path = download_youtube(url, youtube_cache)
                if not src_path:
                    print(f" - fail {vid}: youtube download")
                    continue
            else:
                ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
                if ext not in {".mp4", ".mov", ".mkv", ".avi"}:
                    ext = ".mp4"
                src_candidate = direct_cache / f"{vid}{ext}"
                ok = download_direct(url, src_candidate, timeout=args.timeout)
                if not ok:
                    print(f" - fail {vid}: direct download")
                    continue
                src_path = src_candidate

            dst_path = label_dir / f"{vid}.mp4"
            if dst_path.exists():
                frame_count = count_frames(dst_path)
                if frame_count >= args.min_frames:
                    saved += 1
                    print(f" - keep {vid}: already exists ({frame_count} frames)")
                    continue
                dst_path.unlink(missing_ok=True)

            ok = copy_or_trim(
                src_video=src_path,
                dst_video=dst_path,
                frame_start=frame_start,
                frame_end=frame_end,
                min_frames=args.min_frames,
            )
            if not ok:
                dst_path.unlink(missing_ok=True)
                print(f" - fail {vid}: clip extract")
                continue

            saved += 1
            print(f" - saved {vid} -> {dst_path.name}")

        print(f"[{gloss}] saved {saved} clips (attempted {attempts})")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
