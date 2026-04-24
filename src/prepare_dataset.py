import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .landmarks import (
    BASE_LANDMARK_DIM,
    TwoHandLandmarkExtractor,
    preprocess_landmark_sequence,
)
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed-length sequences for ASL training.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Raw dataset directory (class folders with videos).")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed data.")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="landmarks",
        choices=["landmarks", "rgb"],
        help="Output modality. 'landmarks' is recommended for robust live inference.",
    )
    parser.add_argument("--sequence-length", type=int, default=32, help="Number of frames per sequence.")
    parser.add_argument("--image-size", type=int, default=128, help="Square frame resolution.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Read one every N frames from source video (>=1).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio per class.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=0,
        help="Extra augmented copies per train clip (landmark mode only).",
    )
    parser.add_argument(
        "--landmark-smooth-alpha",
        type=float,
        default=0.65,
        help="EMA alpha for temporal landmark smoothing (0 disables).",
    )
    parser.add_argument(
        "--min-hand-presence",
        type=float,
        default=0.20,
        help="Minimum fraction of frames with at least one detected hand (landmark mode).",
    )
    parser.add_argument(
        "--min-landmark-energy",
        type=float,
        default=0.0008,
        help="Minimum average absolute motion in base landmark features (landmark mode).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".mp4,.avi,.mov,.mkv",
        help="Comma-separated video extensions.",
    )
    return parser.parse_args()


def list_videos(class_dir: Path, extensions: Sequence[str]) -> List[Path]:
    ext_set = {ext.lower() for ext in extensions}
    videos = [p for p in sorted(class_dir.iterdir()) if p.is_file() and p.suffix.lower() in ext_set]
    return videos


def split_items(
    items: Sequence[Path], val_ratio: float, test_ratio: float, rng: random.Random
) -> Tuple[List[Path], List[Path], List[Path]]:
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Invalid split ratios. Require 0 <= val_ratio, test_ratio and val_ratio+test_ratio < 1.")

    items = list(items)
    rng.shuffle(items)
    n = len(items)
    if n < 3:
        return items, [], []

    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def sample_uniform_indices(frame_count: int, sequence_length: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("Video has no frames.")
    return np.linspace(0, frame_count - 1, num=sequence_length, dtype=np.int64)


def augment_landmark_features(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = features.astype(np.float32, copy=True)
    base = BASE_LANDMARK_DIM
    if out.shape[1] < base:
        return out

    coord_noise = rng.normal(0.0, 0.012, size=out[:, :base].shape).astype(np.float32)
    out[:, :base] += coord_noise
    if out.shape[1] >= 2 * base:
        out[:, base : (2 * base)] += 0.5 * coord_noise

    gain = float(rng.uniform(0.92, 1.08))
    out[:, :base] *= gain
    if out.shape[1] >= 2 * base:
        out[:, base : (2 * base)] *= gain

    shift = int(rng.integers(-2, 3))
    if shift > 0:
        out = np.concatenate([out[shift:], np.repeat(out[-1:], shift, axis=0)], axis=0)
    elif shift < 0:
        k = -shift
        out = np.concatenate([np.repeat(out[:1], k, axis=0), out[:-k]], axis=0)

    return out.astype(np.float32, copy=False)


def read_video_frames(video_path: Path, frame_step: int = 1) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, frame_step) == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return frames


def extract_rgb_features(
    video_path: Path,
    sequence_length: int,
    image_size: int,
    frame_step: int = 1,
) -> np.ndarray:
    raw_frames = read_video_frames(video_path, frame_step=frame_step)

    frames: List[np.ndarray] = []
    for frame in raw_frames:
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    frame_stack = np.stack(frames, axis=0)
    indices = sample_uniform_indices(frame_stack.shape[0], sequence_length)
    sampled = frame_stack[indices]
    return sampled.astype(np.uint8)


def extract_landmark_features(
    video_path: Path,
    extractor: TwoHandLandmarkExtractor,
    sequence_length: int,
    frame_step: int = 1,
    smooth_alpha: float = 0.65,
) -> Tuple[np.ndarray, float, float]:
    raw_frames = read_video_frames(video_path, frame_step=frame_step)
    raw_sequence = []
    for frame in raw_frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_sequence.append(extractor.extract(rgb))
    if not raw_sequence:
        raise RuntimeError(f"No landmark frames extracted from: {video_path}")

    landmarks = np.stack(raw_sequence, axis=0).astype(np.float32)
    presence = (np.linalg.norm(landmarks, axis=(2, 3)) > 1e-6).any(axis=1).astype(np.float32)
    presence_ratio = float(np.mean(presence)) if presence.size > 0 else 0.0

    features = preprocess_landmark_sequence(
        landmarks,
        target_length=sequence_length,
        include_velocity=True,
        include_presence=True,
        smooth_alpha=smooth_alpha,
    )
    base = features[:, :BASE_LANDMARK_DIM]
    deltas = np.diff(base, axis=0)
    motion_energy = float(np.mean(np.abs(deltas))) if deltas.size > 0 else 0.0
    return features.astype(np.float32, copy=False), presence_ratio, motion_energy


def iter_class_dirs(raw_dir: Path) -> Iterable[Path]:
    for p in sorted(raw_dir.iterdir()):
        if p.is_dir():
            yield p


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not raw_dir.exists():
        raise ValueError(f"Raw directory does not exist: {raw_dir}")
    seq_root = output_dir / "sequences"
    ensure_dir(str(seq_root))

    extensions = tuple(ext.strip().lower() for ext in args.extensions.split(",") if ext.strip())
    if not extensions:
        raise ValueError("No valid extensions provided.")

    class_dirs = list(iter_class_dirs(raw_dir))
    if not class_dirs:
        raise ValueError(f"No class directories found in {raw_dir}")

    labels = [d.name for d in class_dirs]
    label_map = {label: idx for idx, label in enumerate(labels)}
    save_json(str(output_dir / "label_map.json"), label_map)

    rows = []
    skipped = []
    split_map = {"train": 0, "val": 0, "test": 0}
    rng_np = np.random.default_rng(args.seed)
    extractor = TwoHandLandmarkExtractor() if args.feature_type == "landmarks" else None

    try:
        for class_dir in class_dirs:
            label = class_dir.name
            videos = list_videos(class_dir, extensions)
            if not videos:
                skipped.append(f"{label}: no videos")
                continue

            train_v, val_v, test_v = split_items(videos, args.val_ratio, args.test_ratio, rng)
            by_split = {"train": train_v, "val": val_v, "test": test_v}

            for split, split_videos in by_split.items():
                split_dir = seq_root / split / label
                ensure_dir(str(split_dir))
                for video_path in tqdm(split_videos, desc=f"{label}:{split}", leave=False):
                    try:
                        if args.feature_type == "rgb":
                            payload = extract_rgb_features(
                                video_path,
                                sequence_length=args.sequence_length,
                                image_size=args.image_size,
                                frame_step=args.frame_step,
                            )
                            modality = "rgb"
                            presence_ratio = 1.0
                            motion_energy = 1.0
                        else:
                            if extractor is None:
                                raise RuntimeError("Landmark extractor was not initialized.")
                            payload, presence_ratio, motion_energy = extract_landmark_features(
                                video_path,
                                extractor=extractor,
                                sequence_length=args.sequence_length,
                                frame_step=args.frame_step,
                                smooth_alpha=args.landmark_smooth_alpha,
                            )
                            if presence_ratio < args.min_hand_presence:
                                skipped.append(
                                    f"{video_path}: hand_presence={presence_ratio:.3f} < {args.min_hand_presence:.3f}"
                                )
                                continue
                            if motion_energy < args.min_landmark_energy:
                                skipped.append(
                                    f"{video_path}: motion_energy={motion_energy:.5f} < {args.min_landmark_energy:.5f}"
                                )
                                continue
                            modality = "landmarks"
                    except Exception as exc:  # noqa: BLE001
                        skipped.append(f"{video_path}: {exc}")
                        continue

                    out_file = split_dir / f"{video_path.stem}.npz"
                    if modality == "rgb":
                        np.savez_compressed(out_file, frames=payload)
                    else:
                        np.savez_compressed(out_file, features=payload)

                    rows.append(
                        {
                            "path": str(out_file.resolve()),
                            "label": label,
                            "label_id": label_map[label],
                            "split": split,
                            "feature_type": modality,
                            "hand_presence": f"{presence_ratio:.5f}",
                            "motion_energy": f"{motion_energy:.6f}",
                        }
                    )
                    split_map[split] += 1

                    if (
                        args.feature_type == "landmarks"
                        and split == "train"
                        and args.augment_copies > 0
                        and modality == "landmarks"
                    ):
                        for aug_idx in range(1, args.augment_copies + 1):
                            aug = augment_landmark_features(payload, rng_np)
                            aug_file = split_dir / f"{video_path.stem}_aug{aug_idx:02d}.npz"
                            np.savez_compressed(aug_file, features=aug)
                            rows.append(
                                {
                                    "path": str(aug_file.resolve()),
                                    "label": label,
                                    "label_id": label_map[label],
                                    "split": split,
                                    "feature_type": "landmarks",
                                    "hand_presence": f"{presence_ratio:.5f}",
                                    "motion_energy": f"{motion_energy:.6f}",
                                }
                            )
                            split_map[split] += 1
    finally:
        if extractor is not None:
            extractor.close()

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label", "label_id", "split", "feature_type", "hand_presence", "motion_energy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Dataset preparation complete.")
    print(f"Feature type: {args.feature_type}")
    print(f"Metadata: {metadata_path}")
    print(f"Label map: {output_dir / 'label_map.json'}")
    print(f"Counts: train={split_map['train']} val={split_map['val']} test={split_map['test']}")
    if skipped:
        print(f"Skipped items: {len(skipped)}")
        for msg in skipped[:20]:
            print(f" - {msg}")
        if len(skipped) > 20:
            print(f" - ... {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
