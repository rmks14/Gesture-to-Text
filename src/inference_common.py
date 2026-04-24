from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD, invert_label_map, normalize_landmark_features
from .landmarks import preprocess_landmark_sequence
from .model import SequenceClassifier


def preprocess_frame_rgb(frame_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    frame = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(frame).float() / 255.0
    x = x.permute(2, 0, 1).contiguous()
    x = (x - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
    return x


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    input_type_arg: str,
) -> Tuple[SequenceClassifier, Dict[int, str], str]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    label_map = ckpt["label_map"]
    idx_to_label = invert_label_map(label_map)
    h = ckpt.get("model_hparams", {})

    ckpt_input_type = str(h.get("input_type", "rgb"))
    resolved_input_type = input_type_arg if input_type_arg != "auto" else ckpt_input_type
    input_dim = int(h.get("input_dim", 0)) if resolved_input_type == "landmarks" else 0

    model = SequenceClassifier(
        num_classes=len(label_map),
        input_type=resolved_input_type,
        input_dim=input_dim if resolved_input_type == "landmarks" else None,
        embedding_dim=int(h.get("embedding_dim", 256)),
        hidden_size=int(h.get("hidden_size", 256)),
        num_layers=int(h.get("num_layers", 2)),
        dropout=float(h.get("dropout", 0.3)),
        bidirectional=bool(h.get("bidirectional", False)),
        encoder_type=str(h.get("encoder_type", "auto")),
        pretrained_encoder=False,
        use_attention=bool(h.get("use_attention", True)),
        temporal_conv=bool(h.get("temporal_conv", False)),
        temporal_conv_layers=int(h.get("temporal_conv_layers", 2)),
        temporal_conv_kernel_size=int(h.get("temporal_conv_kernel_size", 5)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, idx_to_label, resolved_input_type


def count_detected_hands(landmarks: np.ndarray) -> int:
    return int(np.sum(np.max(np.abs(landmarks), axis=(1, 2)) > 1e-6))


def forward_probs(
    model: SequenceClassifier,
    input_type: str,
    rgb_sequence: Optional[Deque[torch.Tensor]],
    landmark_sequence: Optional[Deque[np.ndarray]],
    seq_target: int,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    if input_type == "landmarks":
        if landmark_sequence is None:
            raise RuntimeError("landmark_sequence is not initialized.")
        raw = np.stack(list(landmark_sequence), axis=0).astype(np.float32)
        feats = preprocess_landmark_sequence(
            raw,
            target_length=seq_target,
            include_velocity=True,
            include_presence=True,
            smooth_alpha=0.65,
        )
        x = torch.from_numpy(feats).float()
        x = normalize_landmark_features(x)
        x = x.unsqueeze(0).to(device)
    else:
        if rgb_sequence is None:
            raise RuntimeError("rgb_sequence is not initialized.")
        x = torch.stack(list(rgb_sequence), dim=0).unsqueeze(0).to(device)

    temp = max(1e-6, float(temperature))
    with torch.no_grad():
        logits = model(x) / temp
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def topk_probs(probs: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if probs.size == 0:
        return []
    k = max(1, min(int(k), int(probs.shape[0])))
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]
