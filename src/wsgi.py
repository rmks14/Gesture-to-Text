from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path

from .web_app import WebSignRecognizer, create_app


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_checkpoint_path() -> str:
    env_path = os.environ.get("CHECKPOINT_PATH")
    if env_path:
        return env_path

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path("/app/runs/asl_landmarks_10cls_v2_light/best.pt"),
        project_root / "runs" / "asl_landmarks_10cls_v2_light" / "best.pt",
        project_root / "runs" / "asl_landmarks_9cls_v1" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _build_args_from_env() -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=_default_checkpoint_path(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8000),
        input_type=os.environ.get("INPUT_TYPE", "landmarks"),
        sequence_length=_env_int("SEQUENCE_LENGTH", 32),
        image_size=_env_int("IMAGE_SIZE", 128),
        predict_every=_env_int("PREDICT_EVERY", 2),
        smoothing_window=_env_int("SMOOTHING_WINDOW", 8),
        min_confidence=_env_float("MIN_CONFIDENCE", 0.55),
        min_margin=_env_float("MIN_MARGIN", 0.12),
        stable_window=_env_int("STABLE_WINDOW", 5),
        stable_votes=_env_int("STABLE_VOTES", 3),
        min_stable_seconds=_env_float("MIN_STABLE_SECONDS", 1.6),
        top_k=_env_int("TOP_K", 5),
        nlp_max_words=_env_int("NLP_MAX_WORDS", 10),
        nlp_cooldown_seconds=_env_float("NLP_COOLDOWN_SECONDS", 1.1),
        nosign_label=os.environ.get("NOSIGN_LABEL", "no_sign"),
        nosign_threshold=_env_float("NOSIGN_THRESHOLD", 0.50),
        allow_zero_hands=_env_bool("ALLOW_ZERO_HANDS", False),
        device=os.environ.get("DEVICE", "auto"),
    )


_recognizer = WebSignRecognizer(_build_args_from_env())
app = create_app(_recognizer)


@atexit.register
def _close_recognizer() -> None:
    _recognizer.close()
