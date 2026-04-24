from __future__ import annotations

import argparse
import base64
import time
from collections import Counter, deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from .inference_common import (
    count_detected_hands,
    forward_probs,
    load_model_from_checkpoint,
    preprocess_frame_rgb,
    topk_probs,
)
from .landmarks import TwoHandLandmarkExtractor
from .nlp_sentence import OnlineSentenceBuilder
from .utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web UI for ASL sign recognition.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (best.pt).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input-type", type=str, default="auto", choices=["auto", "rgb", "landmarks"])
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--predict-every", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-margin", type=float, default=0.12)
    parser.add_argument("--stable-window", type=int, default=5)
    parser.add_argument("--stable-votes", type=int, default=3)
    parser.add_argument(
        "--min-stable-seconds",
        type=float,
        default=1.6,
        help="Minimum time a confident candidate must persist before being accepted.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nlp-max-words", type=int, default=10)
    parser.add_argument("--nlp-cooldown-seconds", type=float, default=1.1)
    parser.add_argument("--nosign-label", type=str, default="no_sign")
    parser.add_argument("--nosign-threshold", type=float, default=0.50)
    parser.add_argument(
        "--allow-zero-hands",
        action="store_true",
        help="Allow lexical predictions even when no hand landmarks are detected.",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def decode_data_url_image(data_url: str) -> np.ndarray:
    if not data_url:
        raise ValueError("Missing image payload.")
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image.")
    return frame


def serialize_landmarks(landmarks: Optional[np.ndarray]) -> List[Dict]:
    if landmarks is None:
        return []
    out: List[Dict] = []
    for hand_idx in range(landmarks.shape[0]):
        hand = landmarks[hand_idx]
        if np.max(np.abs(hand)) <= 1e-6:
            continue
        points = [[float(p[0]), float(p[1]), float(p[2])] for p in hand]
        out.append({"hand_index": int(hand_idx), "points": points})
    return out


class WebSignRecognizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.model, self.idx_to_label, self.input_type = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=self.device,
            input_type_arg=args.input_type,
        )
        self.label_to_idx = {name: idx for idx, name in self.idx_to_label.items()}
        self.nosign_idx = self.label_to_idx.get(args.nosign_label)
        self.nosign_threshold = float(args.nosign_threshold)
        self.require_hands_for_sign = not bool(args.allow_zero_hands)
        self.landmark_extractor = TwoHandLandmarkExtractor() if self.input_type == "landmarks" else None
        self._lock = Lock()

        self.sequence_length = max(2, int(args.sequence_length))
        self.predict_every = max(1, int(args.predict_every))
        self.smoothing_window = max(1, int(args.smoothing_window))
        self.stable_window = max(1, int(args.stable_window))
        self.stable_votes = max(1, int(args.stable_votes))
        self.min_stable_seconds = max(0.0, float(args.min_stable_seconds))
        self.min_confidence = float(args.min_confidence)
        self.min_margin = float(args.min_margin)
        self.top_k = max(1, int(args.top_k))
        self.image_size = int(args.image_size)

        self.sentence_builder = OnlineSentenceBuilder(
            max_words=args.nlp_max_words,
            cooldown_seconds=args.nlp_cooldown_seconds,
        )
        self._reset_state(clear_words=True)

    def close(self) -> None:
        if self.landmark_extractor is not None:
            self.landmark_extractor.close()

    def _reset_state(self, clear_words: bool) -> None:
        self.rgb_sequence: Optional[Deque[torch.Tensor]] = (
            deque(maxlen=self.sequence_length) if self.input_type == "rgb" else None
        )
        self.landmark_sequence: Optional[Deque[np.ndarray]] = (
            deque(maxlen=self.sequence_length) if self.input_type == "landmarks" else None
        )
        self.proba_buffer: Deque[np.ndarray] = deque(maxlen=self.smoothing_window)
        self.pred_history: Deque[str] = deque(maxlen=self.stable_window)
        self.recent_guesses: Deque[str] = deque(maxlen=30)
        self.frame_idx = 0
        self.last_prediction = "..."
        self.last_confidence = 0.0
        self.last_status = "idle"
        self.last_hands_detected = 0
        self.last_landmarks: List[Dict] = []
        self.last_nosign_prob = 0.0
        self.pending_label = ""
        self.pending_since = 0.0
        self.pending_hold_elapsed = 0.0
        self.last_top_items: List[Tuple[str, float]] = []
        if clear_words:
            self.sentence_builder.clear()

    def reset(self) -> None:
        with self._lock:
            self._reset_state(clear_words=True)

    def stop(self) -> None:
        with self._lock:
            self._reset_state(clear_words=False)

    def clear_history(self) -> None:
        with self._lock:
            self.recent_guesses.clear()
            self.pred_history.clear()
            self.proba_buffer.clear()
            self.sentence_builder.clear()
            self.last_prediction = "..."
            self.last_confidence = 0.0
            self.last_status = "cleared"
            self.last_top_items = []
            self.last_nosign_prob = 0.0
            self.pending_label = ""
            self.pending_since = 0.0
            self.pending_hold_elapsed = 0.0

    def _forward_probs(self) -> np.ndarray:
        return forward_probs(
            model=self.model,
            input_type=self.input_type,
            rgb_sequence=self.rgb_sequence,
            landmark_sequence=self.landmark_sequence,
            seq_target=self.sequence_length,
            device=self.device,
            temperature=1.0,
        )

    def _snapshot(self, current_len: int) -> Dict:
        return {
            "prediction": self.last_prediction,
            "confidence": float(self.last_confidence),
            "status": self.last_status,
            "sequence_length": int(current_len),
            "sequence_target": int(self.sequence_length),
            "top": [{"label": label, "prob": float(prob)} for label, prob in self.last_top_items],
            "guesses": list(self.recent_guesses),
            "words": self.sentence_builder.words(),
            "sentence_preview": self.sentence_builder.sentence(),
            "input_type": self.input_type,
            "hands_detected": int(self.last_hands_detected),
            "landmarks": self.last_landmarks,
            "nosign_prob": float(self.last_nosign_prob),
            "hold_seconds": float(self.pending_hold_elapsed),
            "hold_target_seconds": float(self.min_stable_seconds),
        }

    def predict_frame(self, frame_bgr: np.ndarray) -> Dict:
        with self._lock:
            raw_landmarks = None
            if self.input_type == "landmarks":
                if self.landmark_sequence is None or self.landmark_extractor is None:
                    raise RuntimeError("Landmark pipeline is not initialized.")
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw_landmarks = self.landmark_extractor.extract(rgb)
                self.landmark_sequence.append(raw_landmarks)
                current_len = len(self.landmark_sequence)
                self.last_hands_detected = count_detected_hands(raw_landmarks)
                self.last_landmarks = serialize_landmarks(raw_landmarks)
            else:
                if self.rgb_sequence is None:
                    raise RuntimeError("RGB pipeline is not initialized.")
                self.rgb_sequence.append(preprocess_frame_rgb(frame_bgr, self.image_size))
                current_len = len(self.rgb_sequence)
                self.last_hands_detected = 0
                self.last_landmarks = []

            self.frame_idx += 1
            ready = current_len == self.sequence_length
            classify_tick = self.frame_idx % self.predict_every == 0

            if not ready:
                self.last_status = "warming_up"
                self.last_nosign_prob = 0.0
                self.pending_label = ""
                self.pending_since = 0.0
                self.pending_hold_elapsed = 0.0
                return self._snapshot(current_len=current_len)

            if classify_tick:
                if self.input_type == "landmarks" and self.require_hands_for_sign and self.last_hands_detected <= 0:
                    self.pending_label = ""
                    self.pending_since = 0.0
                    self.pending_hold_elapsed = 0.0
                    self.last_prediction = "..."
                    self.last_confidence = 0.0
                    self.last_nosign_prob = 0.0
                    self.last_status = "no_hands"
                    self.last_top_items = []
                    self.pred_history.append("__uncertain__")
                    return self._snapshot(current_len=current_len)

                probs = self._forward_probs()
                self.proba_buffer.append(probs)
                smoothed = np.mean(np.stack(self.proba_buffer, axis=0), axis=0)
                tk = topk_probs(smoothed, max(3, self.top_k))
                top_idx, top_conf = tk[0]
                second_conf = tk[1][1] if len(tk) > 1 else 0.0
                margin = top_conf - second_conf
                self.last_nosign_prob = float(smoothed[self.nosign_idx]) if self.nosign_idx is not None else 0.0
                suppress_for_nosign = (
                    self.nosign_idx is not None and self.last_nosign_prob >= self.nosign_threshold
                )
                suppress_for_hands = (
                    self.input_type == "landmarks"
                    and self.require_hands_for_sign
                    and self.last_hands_detected <= 0
                )
                is_confident = top_conf >= self.min_confidence and margin >= self.min_margin
                candidate = self.idx_to_label.get(top_idx, f"class_{top_idx}")
                now = time.time()

                if is_confident:
                    if self.nosign_idx is not None and top_idx == self.nosign_idx:
                        is_confident = False
                        self.last_status = "no_sign"
                    if suppress_for_nosign:
                        is_confident = False
                        self.last_status = "no_sign"
                    if suppress_for_hands:
                        is_confident = False
                        self.last_status = "no_hands"

                self.pred_history.append(candidate if is_confident else "__uncertain__")
                votes = Counter(self.pred_history)
                if is_confident:
                    if candidate != self.pending_label:
                        self.pending_label = candidate
                        self.pending_since = now
                    self.pending_hold_elapsed = max(0.0, now - self.pending_since)
                    hold_ready = self.pending_hold_elapsed >= self.min_stable_seconds

                    self.last_prediction = candidate
                    self.last_confidence = float(top_conf)
                    if votes[candidate] >= self.stable_votes and hold_ready:
                        self.last_status = "stable"
                        if self.sentence_builder.add_label(candidate, ts=now):
                            self.recent_guesses.append(candidate)
                    elif votes[candidate] >= self.stable_votes:
                        self.last_status = "holding"
                    else:
                        self.last_status = "stabilizing"
                else:
                    self.pending_label = ""
                    self.pending_since = 0.0
                    self.pending_hold_elapsed = 0.0
                    self.last_prediction = "..."
                    self.last_confidence = float(top_conf)
                    if self.last_status not in {"no_sign", "no_hands"}:
                        self.last_status = "uncertain"

                self.last_top_items = [(self.idx_to_label.get(i, f"class_{i}"), p) for i, p in tk[: self.top_k]]
            else:
                if self.last_status not in {"stable", "uncertain"}:
                    self.last_status = "tracking"

            return self._snapshot(current_len=current_len)

    def generate_sentence(self) -> str:
        with self._lock:
            return self.sentence_builder.sentence()


def create_app(recognizer: WebSignRecognizer) -> Flask:
    here = Path(__file__).resolve().parent
    templates = here / "web" / "templates"
    static = here / "web" / "static"
    app = Flask(__name__, template_folder=str(templates), static_folder=str(static))

    @app.get("/")
    def index() -> str:
        labels = [recognizer.idx_to_label[i] for i in sorted(recognizer.idx_to_label.keys())]
        return render_template(
            "index.html",
            labels=labels,
            input_type=recognizer.input_type,
        )

    @app.get("/api/health")
    def api_health():
        return jsonify(
            {
                "ok": True,
                "status": "healthy",
                "input_type": recognizer.input_type,
                "labels": len(recognizer.idx_to_label),
            }
        )

    @app.post("/api/start")
    def api_start():
        recognizer.reset()
        return jsonify({"ok": True, "status": "started"})

    @app.post("/api/stop")
    def api_stop():
        recognizer.stop()
        return jsonify({"ok": True, "status": "stopped"})

    @app.post("/api/clear")
    def api_clear():
        recognizer.clear_history()
        return jsonify({"ok": True, "status": "cleared"})

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True) or {}
        data_url = str(payload.get("image", ""))
        try:
            frame = decode_data_url_image(data_url)
            result = recognizer.predict_frame(frame)
            return jsonify({"ok": True, **result})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/generate_sentence")
    def api_generate_sentence():
        sentence = recognizer.generate_sentence()
        return jsonify({"ok": True, "sentence": sentence})

    return app


def main() -> None:
    args = parse_args()
    recognizer = WebSignRecognizer(args)
    app = create_app(recognizer)
    print(f"Loaded model from: {args.checkpoint}")
    print(f"Input mode: {recognizer.input_type}")
    print(f"Serving web UI on http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        recognizer.close()


if __name__ == "__main__":
    main()
