import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command pipeline for ASL demo training.")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--run-dir", type=str, default="runs/asl_cnn_lstm_demo")
    parser.add_argument("--feature-type", type=str, default="landmarks", choices=["landmarks", "rgb"])
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise ValueError(f"Raw directory not found: {raw_dir}. Collect data first.")

    run_cmd(
        [
            sys.executable,
            "-m",
            "src.prepare_dataset",
            "--raw-dir",
            args.raw_dir,
            "--output-dir",
            args.processed_dir,
            "--feature-type",
            args.feature_type,
            "--sequence-length",
            str(args.sequence_length),
            "--image-size",
            str(args.image_size),
            "--augment-copies",
            str(args.augment_copies),
        ]
    )

    metadata = str(Path(args.processed_dir) / "metadata.csv")
    label_map = str(Path(args.processed_dir) / "label_map.json")

    run_cmd(
        [
            sys.executable,
            "-m",
            "src.train",
            "--data-index",
            metadata,
            "--label-map",
            label_map,
            "--output-dir",
            args.run_dir,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--input-type",
            args.feature_type,
            "--bidirectional",
            "--balanced-sampler",
            "--class-weighting",
            "--label-smoothing",
            "0.03",
            "--grad-clip",
            "1.0",
            "--lr-scheduler",
            "plateau",
            "--device",
            args.device,
        ]
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--data-index",
            metadata,
            "--label-map",
            label_map,
            "--checkpoint",
            str(Path(args.run_dir) / "best.pt"),
            "--split",
            "test",
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--input-type",
            args.feature_type,
            "--device",
            args.device,
        ]
    )

    print("\nPipeline complete.")
    print(f"Best checkpoint: {Path(args.run_dir) / 'best.pt'}")
    print("Run live inference with:")
    print(
        f"{sys.executable} -m src.live_infer --checkpoint {Path(args.run_dir) / 'best.pt'} "
        f"--input-type {args.feature_type} --sequence-length {args.sequence_length} --image-size {args.image_size}"
    )


if __name__ == "__main__":
    main()
