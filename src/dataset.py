import csv
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .landmarks import BASE_LANDMARK_DIM
from .utils import load_json


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class SequenceDataset(Dataset):
    def __init__(
        self,
        index_file: str,
        split: str,
        normalize: bool = True,
        augment: bool = False,
    ) -> None:
        self.normalize = normalize
        self.augment = augment
        self.samples: List[Tuple[str, int]] = []
        with open(index_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append((row["path"], int(row["label_id"])))
        if not self.samples:
            raise ValueError(f"No samples found for split='{split}' in {index_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        payload = np.load(path)
        if "features" in payload:
            feats = payload["features"]  # (T, F), float32
            x = torch.from_numpy(feats).float()
            if self.augment:
                x = apply_landmark_augmentation(x)
            if self.normalize:
                x = normalize_landmark_features(x)
        elif "frames" in payload:
            frames = payload["frames"]  # (T, H, W, C), uint8
            x = torch.from_numpy(frames).float() / 255.0
            x = x.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
            if self.augment:
                x = apply_augmentation(x)
            if self.normalize:
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
        else:
            raise ValueError(f"Unsupported sample payload keys at {path}. Expected 'frames' or 'features'.")
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class SyntheticNoSignLandmarkDataset(Dataset):
    """Synthetic idle/non-sign landmark sequences used as hard negatives."""

    def __init__(
        self,
        num_samples: int,
        sequence_length: int,
        feature_dim: int,
        label_id: int,
        normalize: bool = True,
        seed: int = 42,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if sequence_length <= 1:
            raise ValueError("sequence_length must be > 1")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        self.num_samples = int(num_samples)
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.label_id = int(label_id)
        self.normalize = bool(normalize)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _empty_features(self) -> torch.Tensor:
        return torch.zeros((self.sequence_length, self.feature_dim), dtype=torch.float32)

    def _has_velocity(self) -> bool:
        return self.feature_dim >= (BASE_LANDMARK_DIM * 2)

    def _has_presence(self) -> bool:
        return self.feature_dim >= (BASE_LANDMARK_DIM * 2 + 2)

    def _rand(self, idx: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed + idx * 104729 + 17)
        return g

    def _build_pattern(self, idx: int) -> torch.Tensor:
        g = self._rand(idx)
        x = self._empty_features()
        base = BASE_LANDMARK_DIM
        has_vel = self._has_velocity()
        has_presence = self._has_presence()

        pattern = idx % 4
        coords = x[:, :base]

        if pattern == 0:
            # Mostly empty frame stream with tiny sensor jitter.
            coords += torch.randn(coords.shape, generator=g, dtype=torch.float32) * 0.002
        elif pattern == 1:
            # One hand present, nearly static (idle hand pose).
            coords[:, : base // 2] = torch.randn((self.sequence_length, base // 2), generator=g) * 0.01
            drift = torch.cumsum(torch.randn((self.sequence_length, base // 2), generator=g) * 0.0008, dim=0)
            coords[:, : base // 2] += drift
        elif pattern == 2:
            # Two hands low-amplitude random walk (non-lexical motion).
            walk = torch.cumsum(torch.randn((self.sequence_length, base), generator=g) * 0.0015, dim=0)
            coords[:] = walk
            coords += torch.randn(coords.shape, generator=g, dtype=torch.float32) * 0.004
        else:
            # Hands entering/exiting without stable sign articulation.
            half = self.sequence_length // 2
            active = torch.randn((half, base), generator=g) * 0.009
            coords[:half] = active
            coords[half:] = torch.randn((self.sequence_length - half, base), generator=g) * 0.002

        if has_vel:
            vel = torch.zeros((self.sequence_length, base), dtype=torch.float32)
            vel[1:] = coords[1:] - coords[:-1]
            x[:, base : (2 * base)] = vel

        if has_presence:
            presence = torch.zeros((self.sequence_length, 2), dtype=torch.float32)
            hand1_energy = torch.mean(torch.abs(coords[:, : base // 2]), dim=1)
            hand2_energy = torch.mean(torch.abs(coords[:, base // 2 :]), dim=1)
            presence[:, 0] = (hand1_energy > 0.0045).float()
            presence[:, 1] = (hand2_energy > 0.0045).float()
            x[:, (2 * base) : (2 * base + 2)] = presence

        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._build_pattern(idx)
        if self.normalize:
            x = normalize_landmark_features(x)
        y = torch.tensor(self.label_id, dtype=torch.long)
        return x, y


def load_label_map(label_map_path: str) -> Dict[str, int]:
    return load_json(label_map_path)


def invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in label_map.items()}


def apply_augmentation(x: torch.Tensor) -> torch.Tensor:
    # Random horizontal flip at sequence level.
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])

    # Random spatial crop-and-resize applied consistently across sequence.
    _, _, h, w = x.shape
    scale = float(torch.empty(1).uniform_(0.85, 1.0).item())
    crop_h = max(1, int(round(h * scale)))
    crop_w = max(1, int(round(w * scale)))
    top = int(torch.randint(0, h - crop_h + 1, (1,)).item())
    left = int(torch.randint(0, w - crop_w + 1, (1,)).item())
    x = x[:, :, top : top + crop_h, left : left + crop_w]
    if crop_h != h or crop_w != w:
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    # Mild photometric jitter.
    brightness = float(torch.empty(1).uniform_(0.85, 1.15).item())
    contrast = float(torch.empty(1).uniform_(0.85, 1.15).item())
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    x = x * brightness
    x = x.clamp(0.0, 1.0)
    return x


def normalize_landmark_features(x: torch.Tensor) -> torch.Tensor:
    # Normalize coordinate/velocity dimensions while leaving hand-presence bits unchanged.
    x = x.clone()
    coord_dims = min(x.shape[1], BASE_LANDMARK_DIM * 2)
    if coord_dims > 0:
        core = x[:, :coord_dims]
        mean = core.mean(dim=0, keepdim=True)
        std = core.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        x[:, :coord_dims] = (core - mean) / std
    return x


def apply_landmark_augmentation(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    coord_dims = min(x.shape[1], BASE_LANDMARK_DIM)
    vel_offset = BASE_LANDMARK_DIM
    has_velocity = x.shape[1] >= (BASE_LANDMARK_DIM * 2)
    has_presence = x.shape[1] >= (BASE_LANDMARK_DIM * 2 + 2)

    if coord_dims > 0:
        noise = torch.randn_like(x[:, :coord_dims]) * 0.015
        x[:, :coord_dims] += noise
        scale = float(torch.empty(1).uniform_(0.92, 1.08).item())
        x[:, :coord_dims] *= scale
        if has_velocity:
            x[:, vel_offset : vel_offset + BASE_LANDMARK_DIM] += 0.5 * noise
            x[:, vel_offset : vel_offset + BASE_LANDMARK_DIM] *= scale

    # Small temporal shift.
    shift = int(torch.randint(-2, 3, (1,)).item())
    if shift > 0:
        x = torch.cat([x[shift:], x[-1:].repeat(shift, 1)], dim=0)
    elif shift < 0:
        k = -shift
        x = torch.cat([x[:1].repeat(k, 1), x[:-k]], dim=0)

    # Randomly hide one hand track to improve robustness to partial detections.
    if torch.rand(1).item() < 0.12:
        hand_idx = int(torch.randint(0, 2, (1,)).item())
        per_hand = BASE_LANDMARK_DIM // 2
        start = hand_idx * per_hand
        end = start + per_hand
        x[:, start:end] = 0.0
        if has_velocity:
            x[:, vel_offset + start : vel_offset + end] = 0.0
        if has_presence:
            x[:, BASE_LANDMARK_DIM * 2 + hand_idx] = 0.0

    return x
