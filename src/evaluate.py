import argparse
from collections import defaultdict
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SequenceDataset, invert_label_map, load_label_map
from .model import SequenceClassifier
from .utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ASL checkpoint.")
    parser.add_argument("--data-index", type=str, required=True, help="Path to metadata.csv")
    parser.add_argument("--label-map", type=str, required=True, help="Path to label_map.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-type", type=str, default="auto", choices=["auto", "rgb", "landmarks"])
    parser.add_argument("--input-dim", type=int, default=0, help="Only needed for legacy checkpoints without metadata.")
    parser.add_argument(
        "--no-prefer-checkpoint-label-map",
        action="store_true",
        help="Disable using checkpoint label_map; fall back to --label-map file.",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def infer_input_type(sample: torch.Tensor, arg_input_type: str) -> str:
    if arg_input_type != "auto":
        return arg_input_type
    if sample.ndim == 4:
        return "rgb"
    if sample.ndim == 2:
        return "landmarks"
    raise ValueError(f"Unsupported sample shape for input type inference: {tuple(sample.shape)}")


def load_model(
    checkpoint: Dict,
    num_classes: int,
    device: torch.device,
    fallback_input_type: str,
    fallback_input_dim: int,
) -> nn.Module:
    hparams = checkpoint.get("model_hparams", {})
    input_type = str(hparams.get("input_type", fallback_input_type))
    input_dim = int(hparams.get("input_dim", fallback_input_dim))

    model = SequenceClassifier(
        num_classes=num_classes,
        input_type=input_type,
        input_dim=input_dim if input_type == "landmarks" else None,
        embedding_dim=int(hparams.get("embedding_dim", 256)),
        hidden_size=int(hparams.get("hidden_size", 256)),
        num_layers=int(hparams.get("num_layers", 2)),
        dropout=float(hparams.get("dropout", 0.3)),
        bidirectional=bool(hparams.get("bidirectional", False)),
        encoder_type=str(hparams.get("encoder_type", "auto")),
        pretrained_encoder=False,
        use_attention=bool(hparams.get("use_attention", True)),
        temporal_conv=bool(hparams.get("temporal_conv", False)),
        temporal_conv_layers=int(hparams.get("temporal_conv_layers", 2)),
        temporal_conv_kernel_size=int(hparams.get("temporal_conv_kernel_size", 5)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    file_label_map = load_label_map(args.label_map)
    dataset = SequenceDataset(index_file=args.data_index, split=args.split, normalize=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_label_map = checkpoint.get("label_map", {})
    if (not args.no_prefer_checkpoint_label_map) and ckpt_label_map:
        label_map = ckpt_label_map
        print("Using label map from checkpoint.")
    else:
        label_map = file_label_map
    idx_to_label = invert_label_map(label_map)

    sample_x, _ = dataset[0]
    inferred_input_type = infer_input_type(sample_x, args.input_type)
    inferred_input_dim = int(args.input_dim or (sample_x.shape[1] if inferred_input_type == "landmarks" else 0))

    model = load_model(
        checkpoint,
        num_classes=len(label_map),
        device=device,
        fallback_input_type=inferred_input_type,
        fallback_input_dim=inferred_input_dim,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    top3_correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for x, y in tqdm(loader, desc=f"eval:{args.split}", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        topk = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
        total_loss += loss.item() * y.size(0)
        correct += (preds == y).sum().item()
        top3_correct += (topk == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)

        for yi, pi in zip(y.tolist(), preds.tolist()):
            class_total[yi] += 1
            if yi == pi:
                class_correct[yi] += 1

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    top3 = top3_correct / max(1, total)
    print(f"{args.split} loss: {avg_loss:.4f}")
    print(f"{args.split} accuracy: {acc:.4f} ({correct}/{total})")
    print(f"{args.split} top-3 accuracy: {top3:.4f} ({top3_correct}/{total})")
    print("Per-class accuracy:")
    for idx in sorted(idx_to_label.keys()):
        cname = idx_to_label[idx]
        c_total = class_total[idx]
        c_acc = class_correct[idx] / c_total if c_total > 0 else 0.0
        print(f" - {cname}: {c_acc:.4f} ({class_correct[idx]}/{c_total})")


if __name__ == "__main__":
    main()
