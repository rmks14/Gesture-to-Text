from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None  # type: ignore[assignment]


NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORD_DIM = 3
BASE_LANDMARK_DIM = NUM_HANDS * LANDMARKS_PER_HAND * COORD_DIM  # 126


@dataclass
class HandsConfig:
    static_image_mode: bool = False
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_hands: int = NUM_HANDS


class TwoHandLandmarkExtractor:
    def __init__(self, config: Optional[HandsConfig] = None) -> None:
        if mp is None:
            raise ImportError(
                "mediapipe is required for landmark extraction. Install with: pip install mediapipe"
            )
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise ImportError(
                "This pipeline requires MediaPipe Hands (`mp.solutions.hands`). "
                "Install pinned version: pip install mediapipe==0.10.14"
            )
        cfg = config or HandsConfig()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=cfg.static_image_mode,
            model_complexity=cfg.model_complexity,
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray:
        # Returns shape (2, 21, 3), zero-padded if one or no hands detected.
        result = self._hands.process(frame_rgb)
        out = np.zeros((NUM_HANDS, LANDMARKS_PER_HAND, COORD_DIM), dtype=np.float32)
        if not result.multi_hand_landmarks:
            return out

        hands = []
        for hand_landmarks in result.multi_hand_landmarks[:NUM_HANDS]:
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            if arr.shape != (LANDMARKS_PER_HAND, COORD_DIM):
                continue
            wrist_x = float(arr[0, 0])
            hands.append((wrist_x, arr))
        if not hands:
            return out

        # Canonical ordering: left-to-right in image space.
        hands.sort(key=lambda x: x[0])
        for idx, (_, arr) in enumerate(hands[:NUM_HANDS]):
            out[idx] = arr
        return out


def temporal_resample(sequence: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("target_length must be > 0")
    t = sequence.shape[0]
    if t == target_length:
        return sequence.astype(np.float32, copy=True)
    if t <= 1:
        return np.repeat(sequence[:1], repeats=target_length, axis=0).astype(np.float32, copy=False)

    old_idx = np.linspace(0.0, 1.0, num=t, dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    flat = sequence.reshape(t, -1)
    out = np.empty((target_length, flat.shape[1]), dtype=np.float32)
    for feat_idx in range(flat.shape[1]):
        out[:, feat_idx] = np.interp(new_idx, old_idx, flat[:, feat_idx])
    return out.reshape((target_length, *sequence.shape[1:]))


def _interpolate_missing_track(track: np.ndarray) -> np.ndarray:
    # track: (T, 21, 3) for one hand; missing frames are all-zero.
    out = track.copy()
    t = out.shape[0]
    flat = out.reshape(t, -1)
    valid = np.any(np.abs(flat) > 1e-8, axis=1)
    if not np.any(valid):
        return out
    if np.all(valid):
        return out

    valid_idx = np.where(valid)[0]
    x = np.arange(t, dtype=np.float32)
    for feat_idx in range(flat.shape[1]):
        feat = flat[:, feat_idx]
        flat[:, feat_idx] = np.interp(x, valid_idx.astype(np.float32), feat[valid_idx])
    return flat.reshape(track.shape)


def fill_missing_hands(sequence: np.ndarray) -> np.ndarray:
    # sequence shape: (T, 2, 21, 3)
    out = sequence.astype(np.float32, copy=True)
    for hand_idx in range(NUM_HANDS):
        out[:, hand_idx] = _interpolate_missing_track(out[:, hand_idx])
    return out


def normalize_hands(sequence: np.ndarray) -> np.ndarray:
    out = sequence.astype(np.float32, copy=True)
    for t in range(out.shape[0]):
        for hand_idx in range(NUM_HANDS):
            hand = out[t, hand_idx]
            if not np.any(np.abs(hand) > 1e-8):
                continue
            wrist = hand[0:1]
            centered = hand - wrist
            scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
            if scale < 1e-4:
                scale = 1.0
            out[t, hand_idx] = centered / scale
    return out


def smooth_sequence(sequence: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    if not (0.0 < alpha <= 1.0):
        return sequence.astype(np.float32, copy=True)
    out = sequence.astype(np.float32, copy=True)
    for t in range(1, out.shape[0]):
        out[t] = alpha * out[t - 1] + (1.0 - alpha) * out[t]
    return out


def sequence_to_features(
    sequence: np.ndarray,
    include_velocity: bool = True,
    include_presence: bool = True,
) -> np.ndarray:
    # sequence: (T, 2, 21, 3)
    base = sequence.reshape(sequence.shape[0], -1).astype(np.float32, copy=False)
    parts = [base]
    if include_velocity:
        vel = np.zeros_like(base, dtype=np.float32)
        vel[1:] = np.diff(base, axis=0)
        parts.append(vel)
    if include_presence:
        presence = (np.linalg.norm(sequence, axis=(2, 3)) > 1e-6).astype(np.float32)
        parts.append(presence)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def preprocess_landmark_sequence(
    sequence: np.ndarray,
    target_length: int,
    include_velocity: bool = True,
    include_presence: bool = True,
    smooth_alpha: float = 0.65,
) -> np.ndarray:
    if sequence.ndim != 4 or sequence.shape[1:] != (NUM_HANDS, LANDMARKS_PER_HAND, COORD_DIM):
        raise ValueError(
            "Expected sequence shape (T, 2, 21, 3), got "
            f"{tuple(sequence.shape)}"
        )
    filled = fill_missing_hands(sequence)
    resampled = temporal_resample(filled, target_length=target_length)
    normalized = normalize_hands(resampled)
    smoothed = smooth_sequence(normalized, alpha=smooth_alpha)
    return sequence_to_features(
        smoothed,
        include_velocity=include_velocity,
        include_presence=include_presence,
    )
