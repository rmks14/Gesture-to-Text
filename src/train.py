import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .dataset import SequenceDataset, SyntheticNoSignLandmarkDataset, load_label_map
from .model import SequenceClassifier
from .utils import ensure_dir, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASL sign classifier (RGB or MediaPipe 2-hand landmarks).")
    parser.add_argument("--data-index", type=str, required=True, help="Path to metadata.csv")
    parser.add_argument("--label-map", type=str, required=True, help="Path to label_map.json")
    parser.add_argument("--output-dir", type=str, default="runs/asl_cnn_lstm", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau", "cosine"])
    parser.add_argument("--early-stop-patience", type=int, default=12, help="Stop if selected val metric does not improve.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-type", type=str, default="auto", choices=["auto", "rgb", "landmarks"])
    parser.add_argument("--input-dim", type=int, default=0, help="Only used for landmark mode. 0 = infer.")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--encoder-type", type=str, default="auto", choices=["auto", "simple_cnn", "resnet18"])
    parser.add_argument("--pretrained-encoder", action="store_true")
    parser.add_argument("--no-attention", action="store_true", help="Disable temporal attention pooling.")
    parser.add_argument(
        "--temporal-conv",
        action="store_true",
        help="Enable lightweight depthwise-separable temporal conv refinement before LSTM.",
    )
    parser.add_argument("--temporal-conv-layers", type=int, default=2)
    parser.add_argument("--temporal-conv-kernel-size", type=int, default=5)
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Path to checkpoint to resume from.")
    parser.add_argument("--class-weighting", action="store_true", help="Use inverse-frequency class weights.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Use weighted random sampler for train split.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max grad norm. 0 disables clipping.")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time sequence augmentation.")
    parser.add_argument("--loss", type=str, default="cross_entropy", choices=["cross_entropy", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=-1.0, help="Set >0 to use alpha scaling in focal loss.")
    parser.add_argument(
        "--model-select",
        type=str,
        default="f1",
        choices=["acc", "f1"],
        help="Validation metric used for best-checkpoint and early-stop decisions.",
    )
    parser.add_argument(
        "--add-nosign-class",
        action="store_true",
        help="Append synthetic landmark negatives as an extra 'no_sign' class (landmark mode only).",
    )
    parser.add_argument("--nosign-label", type=str, default="no_sign")
    parser.add_argument("--nosign-train-ratio", type=float, default=0.40)
    parser.add_argument("--nosign-val-ratio", type=float, default=0.20)
    return parser.parse_args()


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = -1.0,
        class_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = max(0.0, float(gamma))
        self.alpha = float(alpha)
        self.class_weight = class_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt).clamp_min(1e-8) ** self.gamma) * ce

        if self.class_weight is not None:
            focal = focal * self.class_weight[target]
        if self.alpha > 0.0:
            focal = focal * self.alpha
        return focal.mean()


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=preds.device)
    for y, p in zip(labels, preds):
        cm[y, p] += 1
    return cm


def macro_f1_from_cm(cm: torch.Tensor) -> float:
    num_classes = cm.shape[0]
    f1_scores = []
    for i in range(num_classes):
        tp = float(cm[i, i].item())
        fp = float(cm[:, i].sum().item() - tp)
        fn = float(cm[i, :].sum().item() - tp)
        denom = (2.0 * tp + fp + fn)
        f1 = (2.0 * tp / denom) if denom > 0.0 else 0.0
        f1_scores.append(f1)
    return float(sum(f1_scores) / max(1, len(f1_scores)))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    grad_clip: float = 0.0,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        preds = logits.argmax(dim=1)
        cm += _confusion_matrix(preds, y, num_classes=num_classes)
        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, y)
        num_batches += 1

    train_f1 = macro_f1_from_cm(cm.detach().cpu())
    return (
        running_loss / max(1, num_batches),
        running_acc / max(1, num_batches),
        train_f1,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        cm += _confusion_matrix(preds, y, num_classes=num_classes)
        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, y)
        num_batches += 1

    val_f1 = macro_f1_from_cm(cm.detach().cpu())
    return (
        running_loss / max(1, num_batches),
        running_acc / max(1, num_batches),
        val_f1,
    )


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    label_map: Dict[str, int],
    args: argparse.Namespace,
    epoch: int,
    val_acc: float,
    val_f1: float,
    val_score: float,
    input_type: str,
    input_dim: int,
) -> Dict:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "label_map": label_map,
        "model_hparams": {
            "num_classes": len(label_map),
            "input_type": input_type,
            "input_dim": input_dim,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "encoder_type": args.encoder_type,
            "pretrained_encoder": args.pretrained_encoder,
            "use_attention": not args.no_attention,
            "temporal_conv": args.temporal_conv,
            "temporal_conv_layers": args.temporal_conv_layers,
            "temporal_conv_kernel_size": args.temporal_conv_kernel_size,
        },
        "optimizer_state": optimizer.state_dict(),
        "train_hparams": vars(args),
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_score": val_score,
        "model_select": args.model_select,
    }


def labels_from_dataset(dataset: Dataset) -> List[int]:
    if isinstance(dataset, SequenceDataset):
        return [label_id for _, label_id in dataset.samples]
    if isinstance(dataset, SyntheticNoSignLandmarkDataset):
        return [dataset.label_id] * len(dataset)
    if isinstance(dataset, ConcatDataset):
        labels: List[int] = []
        for sub in dataset.datasets:
            labels.extend(labels_from_dataset(sub))
        return labels
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y.item()))
    return labels


def class_counts_from_labels(labels: List[int], num_classes: int) -> Dict[int, int]:
    counts: Dict[int, int] = {i: 0 for i in range(num_classes)}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return counts


def class_weights_from_counts(class_counts: Dict[int, int], num_classes: int) -> torch.Tensor:
    total = float(sum(class_counts.values()))
    weights = []
    for i in range(num_classes):
        c = class_counts.get(i, 0)
        if c <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * float(c)))
    return torch.tensor(weights, dtype=torch.float32)


def infer_input_type(sample: torch.Tensor, arg_input_type: str) -> str:
    if arg_input_type != "auto":
        return arg_input_type
    if sample.ndim == 4:
        return "rgb"
    if sample.ndim == 2:
        return "landmarks"
    raise ValueError(f"Could not infer input type from sample shape {tuple(sample.shape)}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if scheduler_type == "none":
        return None
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir).resolve()
    ensure_dir(str(output_dir))

    label_map = dict(load_label_map(args.label_map))
    base_train_ds = SequenceDataset(index_file=args.data_index, split="train", normalize=True, augment=not args.no_augment)
    try:
        base_val_ds = SequenceDataset(index_file=args.data_index, split="val", normalize=True, augment=False)
    except ValueError:
        try:
            base_val_ds = SequenceDataset(index_file=args.data_index, split="test", normalize=True, augment=False)
            print("No 'val' split found. Falling back to 'test' split for validation metrics.")
        except ValueError:
            base_val_ds = SequenceDataset(index_file=args.data_index, split="train", normalize=True, augment=False)
            print("No 'val' or 'test' split found. Falling back to 'train' split for validation metrics.")

    sample_x, _ = base_train_ds[0]
    input_type = infer_input_type(sample_x, args.input_type)
    input_dim = int(args.input_dim or (sample_x.shape[1] if input_type == "landmarks" else 0))
    if input_type == "rgb":
        print("Input mode: RGB frames")
    else:
        print(f"Input mode: MediaPipe landmarks (feature_dim={input_dim})")

    train_ds: Dataset = base_train_ds
    val_ds: Dataset = base_val_ds

    if args.add_nosign_class:
        if input_type != "landmarks":
            raise ValueError("--add-nosign-class is only supported for landmark input.")
        if args.nosign_label not in label_map:
            label_map[args.nosign_label] = len(label_map)
        nosign_id = int(label_map[args.nosign_label])

        nosign_train_count = int(round(len(base_train_ds) * max(0.0, args.nosign_train_ratio)))
        nosign_val_count = int(round(len(base_val_ds) * max(0.0, args.nosign_val_ratio)))
        seq_len, feat_dim = int(sample_x.shape[0]), int(sample_x.shape[1])

        extra_train_sets: List[Dataset] = [base_train_ds]
        extra_val_sets: List[Dataset] = [base_val_ds]
        if nosign_train_count > 0:
            extra_train_sets.append(
                SyntheticNoSignLandmarkDataset(
                    num_samples=nosign_train_count,
                    sequence_length=seq_len,
                    feature_dim=feat_dim,
                    label_id=nosign_id,
                    normalize=True,
                    seed=args.seed + 103,
                )
            )
        if nosign_val_count > 0:
            extra_val_sets.append(
                SyntheticNoSignLandmarkDataset(
                    num_samples=nosign_val_count,
                    sequence_length=seq_len,
                    feature_dim=feat_dim,
                    label_id=nosign_id,
                    normalize=True,
                    seed=args.seed + 307,
                )
            )

        train_ds = ConcatDataset(extra_train_sets) if len(extra_train_sets) > 1 else extra_train_sets[0]
        val_ds = ConcatDataset(extra_val_sets) if len(extra_val_sets) > 1 else extra_val_sets[0]
        print(
            f"Added synthetic '{args.nosign_label}' class: "
            f"train+={nosign_train_count}, val+={nosign_val_count}."
        )

    pin_memory = device.type == "cuda"
    train_labels = labels_from_dataset(train_ds)
    class_counts = class_counts_from_labels(train_labels, num_classes=len(label_map))
    idx_to_label = {idx: label for label, idx in label_map.items()}
    print("Train class counts:")
    for idx in range(len(label_map)):
        print(f" - {idx_to_label[idx]}: {class_counts.get(idx, 0)}")

    train_sampler = None
    train_shuffle = True
    if args.balanced_sampler:
        sample_weights = [1.0 / max(1, class_counts[label_id]) for label_id in train_labels]
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = SequenceClassifier(
        num_classes=len(label_map),
        input_type=input_type,
        input_dim=input_dim if input_type == "landmarks" else None,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        encoder_type=args.encoder_type,
        pretrained_encoder=args.pretrained_encoder,
        use_attention=not args.no_attention,
        temporal_conv=args.temporal_conv,
        temporal_conv_layers=args.temporal_conv_layers,
        temporal_conv_kernel_size=args.temporal_conv_kernel_size,
    ).to(device)

    class_weight_tensor = None
    if args.class_weighting:
        class_weight_tensor = class_weights_from_counts(class_counts, num_classes=len(label_map)).to(device)
        print("Class weights:")
        for idx in range(len(label_map)):
            print(f" - {idx_to_label[idx]}: {class_weight_tensor[idx].item():.4f}")

    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=args.label_smoothing)
    else:
        if args.label_smoothing > 0.0:
            print("Note: label smoothing is ignored for focal loss.")
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            class_weight=class_weight_tensor,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer=optimizer, scheduler_type=args.lr_scheduler, epochs=args.epochs)

    best_score = -1.0
    best_epoch = 0
    start_epoch = 1
    history = []

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).resolve()
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer_state = ckpt.get("optimizer_state")
        if optimizer_state:
            try:
                optimizer.load_state_dict(optimizer_state)
                print("Loaded optimizer state from checkpoint.")
            except Exception as exc:  # noqa: BLE001
                print(f"Could not load optimizer state: {exc}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("val_score", -1.0))
        best_epoch = int(ckpt.get("epoch", 0))
        print(
            f"Resumed from {resume_path} "
            f"(last_epoch={ckpt.get('epoch', 0)}, best_val_score={best_score:.4f})"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=len(label_map),
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=len(label_map),
        )
        lr = float(optimizer.param_groups[0]["lr"])
        val_score = val_f1 if args.model_select == "f1" else val_acc

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_score": val_score,
            "lr": lr,
        }
        history.append(metrics)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"val_score={val_score:.4f} lr={lr:.2e}"
        )

        ckpt = build_checkpoint(
            model=model,
            optimizer=optimizer,
            label_map=label_map,
            args=args,
            epoch=epoch,
            val_acc=val_acc,
            val_f1=val_f1,
            val_score=val_score,
            input_type=input_type,
            input_dim=input_dim,
        )
        latest_path = output_dir / "latest.pt"
        torch.save(ckpt, latest_path)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_path = output_dir / "best.pt"
            torch.save(ckpt, best_path)
            print(
                f"Saved new best checkpoint: {best_path} "
                f"({args.model_select}={val_score:.4f})"
            )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_score)
            else:
                scheduler.step()

        if epoch >= args.min_epochs and (epoch - best_epoch) >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: no {args.model_select} improvement for "
                f"{args.early_stop_patience} epochs (best={best_score:.4f} @ epoch {best_epoch})."
            )
            break

    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. History saved to {history_path}")


if __name__ == "__main__":
    main()
