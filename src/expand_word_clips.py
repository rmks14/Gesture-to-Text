import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
MSASL_ALIASES = {
    "thank_you": "thanks",
}
WLASL_ALIASES = {
    "thank_you": "thank you",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top up word-level raw clips from MS-ASL and WLASL sources.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw clip root: data/raw/<label>/*.mp4")
    parser.add_argument(
        "--labels",
        type=str,
        default="hello,how,no,ok,thank_you,yes,you,i,fine",
        help="Comma-separated target labels to expand.",
    )
    parser.add_argument("--target-per-label", type=int, default=180, help="Desired minimum clips per label.")
    parser.add_argument("--msasl-info-tar", type=str, default="data/msasl/msasl.tar")
    parser.add_argument("--wlasl-index-file", type=str, default="tmp_wlasl/start_kit/WLASL_v0.3.json")
    parser.add_argument("--skip-msasl", action="store_true")
    parser.add_argument("--skip-wlasl", action="store_true")
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--max-variants-per-item", type=int, default=3)
    parser.add_argument("--jitter-seconds", type=float, default=0.30)
    return parser.parse_args()


def parse_csv(raw: str) -> List[str]:
    out = [x.strip() for x in raw.split(",")]
    out = [x for x in out if x]
    if not out:
        raise ValueError("No labels provided.")
    return out


def count_by_label(raw_dir: Path, labels: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        label_dir = raw_dir / label
        if not label_dir.exists():
            counts[label] = 0
            continue
        clips = [
            p
            for p in label_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
        counts[label] = len(clips)
    return counts


def print_counts(title: str, counts: Dict[str, int]) -> None:
    print(f"\n{title}")
    for label in sorted(counts.keys()):
        print(f" - {label}: {counts[label]}")


def build_msasl_word_map(labels: List[str]) -> str:
    parts = []
    for label in labels:
        src = MSASL_ALIASES.get(label, label.replace("_", " "))
        parts.append(f"{src}:{label}")
    return ",".join(parts)


def build_wlasl_glosses(labels: List[str]) -> str:
    glosses = [WLASL_ALIASES.get(label, label.replace("_", " ")) for label in labels]
    return ",".join(glosses)


def run_cmd(cmd: List[str]) -> None:
    print("\n>>>", " ".join(cmd))
    rv = subprocess.run(cmd, check=False)
    if rv.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {rv.returncode}: {' '.join(cmd)}")


def labels_with_deficit(counts: Dict[str, int], target: int) -> List[str]:
    return [label for label, count in counts.items() if count < target]


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    labels = parse_csv(args.labels)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (raw_dir / label).mkdir(parents=True, exist_ok=True)

    before = count_by_label(raw_dir, labels)
    print_counts("Current clip counts", before)
    need = labels_with_deficit(before, target=args.target_per_label)
    if not need:
        print("\nAll labels already meet target count.")
        return

    if not args.skip_msasl:
        msasl_tar = Path(args.msasl_info_tar).resolve()
        if msasl_tar.exists():
            word_map = build_msasl_word_map(need)
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "src.download_msasl_from_info",
                    "--info-tar",
                    str(msasl_tar),
                    "--word-map",
                    word_map,
                    "--output-dir",
                    str(raw_dir),
                    "--max-per-label",
                    str(args.target_per_label),
                    "--min-frames",
                    str(args.min_frames),
                    "--max-variants-per-item",
                    str(args.max_variants_per_item),
                    "--jitter-seconds",
                    str(args.jitter_seconds),
                ]
            )
        else:
            print(f"\nMS-ASL tar not found: {msasl_tar} (skipping MS-ASL source)")

    mid = count_by_label(raw_dir, labels)
    need = labels_with_deficit(mid, target=args.target_per_label)

    if need and not args.skip_wlasl:
        wlasl_index = Path(args.wlasl_index_file).resolve()
        if wlasl_index.exists():
            glosses = build_wlasl_glosses(need)
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "src.download_wlasl_subset",
                    "--index-file",
                    str(wlasl_index),
                    "--glosses",
                    glosses,
                    "--output-dir",
                    str(raw_dir),
                    "--max-per-label",
                    str(args.target_per_label),
                    "--min-frames",
                    str(args.min_frames),
                ]
            )
        else:
            print(f"\nWLASL index not found: {wlasl_index} (skipping WLASL source)")

    after = count_by_label(raw_dir, labels)
    print_counts("Updated clip counts", after)

    still_missing = labels_with_deficit(after, target=args.target_per_label)
    if still_missing:
        print("\nLabels still below target (use collect_data for manual top-up):")
        for label in still_missing:
            print(f" - {label}: {after[label]}/{args.target_per_label}")
    else:
        print("\nAll labels reached target clip count.")


if __name__ == "__main__":
    main()

