import argparse
import time
from pathlib import Path
from typing import List

import cv2

from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect isolated-sign videos from webcam.")
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Comma-separated sign labels, e.g. hello,thanks,yes,no",
    )
    parser.add_argument("--samples-per-label", type=int, default=20, help="Number of clips to record per label.")
    parser.add_argument("--clip-seconds", type=float, default=2.0, help="Duration of each clip in seconds.")
    parser.add_argument("--fps", type=int, default=20, help="Target capture FPS.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output root directory.")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before each recording.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    return parser.parse_args()


def draw_overlay(
    frame,
    label: str,
    sample_idx: int,
    total_samples: int,
    status: str,
    seconds_left: float | None = None,
) -> None:
    cv2.putText(frame, f"Label: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Sample: {sample_idx}/{total_samples}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, status, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    if seconds_left is not None:
        cv2.putText(
            frame,
            f"{seconds_left:0.1f}s",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "Keys: s=start label, n=next clip, q=quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def sanitize_labels(raw_labels: str) -> List[str]:
    labels = [item.strip() for item in raw_labels.split(",")]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError("No valid labels provided.")
    return labels


def wait_for_key(cap: cv2.VideoCapture, target_key: int, label: str, sample_idx: int, total_samples: int) -> bool:
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        draw_overlay(frame, label, sample_idx, total_samples, "Press 's' to start this label")
        cv2.imshow("ASL Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == target_key:
            return True


def record_clip(
    cap: cv2.VideoCapture,
    out_path: Path,
    label: str,
    sample_idx: int,
    total_samples: int,
    clip_seconds: float,
    fps: int,
    countdown: int,
) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        return False

    start_countdown = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            writer.release()
            return False
        frame = cv2.flip(frame, 1)
        remaining = countdown - (time.time() - start_countdown)
        if remaining <= 0:
            break
        draw_overlay(frame, label, sample_idx, total_samples, "Get into pose", remaining)
        cv2.imshow("ASL Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            writer.release()
            return False

    start_record = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            writer.release()
            return False
        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start_record
        if elapsed >= clip_seconds:
            break

        writer.write(frame)
        remaining = clip_seconds - elapsed
        draw_overlay(frame, label, sample_idx, total_samples, "Recording", remaining)
        cv2.imshow("ASL Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            writer.release()
            return False

    writer.release()
    return True


def main() -> None:
    args = parse_args()
    labels = sanitize_labels(args.labels)
    raw_root = Path(args.output_dir).resolve()
    ensure_dir(str(raw_root))

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Saving clips to: {raw_root}")
    print(f"Labels: {labels}")
    print(f"Samples/label: {args.samples_per_label}, clip_seconds: {args.clip_seconds}, fps: {args.fps}")

    try:
        for label in labels:
            label_dir = raw_root / label
            ensure_dir(str(label_dir))
            print(f"\nCollecting label: {label}")
            ready = wait_for_key(cap, ord("s"), label, 0, args.samples_per_label)
            if not ready:
                break

            for idx in range(1, args.samples_per_label + 1):
                out_path = label_dir / f"{label}_{idx:03d}.mp4"
                ok = record_clip(
                    cap=cap,
                    out_path=out_path,
                    label=label,
                    sample_idx=idx,
                    total_samples=args.samples_per_label,
                    clip_seconds=args.clip_seconds,
                    fps=args.fps,
                    countdown=args.countdown,
                )
                if not ok:
                    print("Collection interrupted.")
                    return
                print(f"Saved: {out_path}")

                while True:
                    frame_ok, frame = cap.read()
                    if not frame_ok:
                        return
                    frame = cv2.flip(frame, 1)
                    draw_overlay(frame, label, idx, args.samples_per_label, "Press 'n' for next clip")
                    cv2.imshow("ASL Data Collection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return
                    if key == ord("n"):
                        break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Data collection finished.")


if __name__ == "__main__":
    main()
