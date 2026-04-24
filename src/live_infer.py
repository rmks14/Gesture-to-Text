import argparse
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

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
    parser = argparse.ArgumentParser(description="Live ASL sign inference (RGB or 2-hand MediaPipe landmarks).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--input-type", type=str, default="auto", choices=["auto", "rgb", "landmarks"])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--camera-backend",
        type=str,
        default="auto",
        choices=["auto", "default", "msmf", "dshow"],
        help="Camera backend on Windows. 'auto' tries multiple backends.",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--predict-every", type=int, default=2, help="Run model every N frames after warmup")
    parser.add_argument("--smoothing-window", type=int, default=8, help="Probability moving-average window")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Threshold to display prediction")
    parser.add_argument("--min-margin", type=float, default=0.12, help="Top1-top2 probability margin threshold.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (>1 softens probs).")
    parser.add_argument(
        "--stable-window",
        type=int,
        default=5,
        help="Recent prediction window used for vote stabilization.",
    )
    parser.add_argument("--stable-votes", type=int, default=3, help="Votes required in stable window.")
    parser.add_argument("--min-motion", type=float, default=0.012, help="Min frame-diff motion score to allow classify.")
    parser.add_argument("--motion-window", type=int, default=6, help="Window for smoothing motion score.")
    parser.add_argument("--idle-reset-frames", type=int, default=12, help="Idle frames before clearing prediction state.")
    parser.add_argument(
        "--strict-labels",
        type=str,
        default="no",
        help="Comma-separated labels requiring stricter confidence (e.g. no).",
    )
    parser.add_argument("--strict-confidence", type=float, default=0.78, help="Confidence threshold for strict labels.")
    parser.add_argument("--strict-margin", type=float, default=0.24, help="Top1-top2 margin threshold for strict labels.")
    parser.add_argument("--nosign-label", type=str, default="no_sign", help="Label name used for no-sign class.")
    parser.add_argument(
        "--nosign-threshold",
        type=float,
        default=0.50,
        help="If P(no_sign) >= threshold, suppress lexical sign output.",
    )
    parser.add_argument(
        "--allow-zero-hands",
        action="store_true",
        help="Allow predictions even when no hands are detected in the latest frame.",
    )
    parser.add_argument(
        "--idle-prior-alpha",
        type=float,
        default=0.97,
        help="EMA alpha for idle prior update (higher = slower updates).",
    )
    parser.add_argument(
        "--debias-strength",
        type=float,
        default=0.70,
        help="How strongly to divide by idle prior probabilities (0 disables).",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classes to show on screen.")
    parser.add_argument("--nlp-max-words", type=int, default=10, help="Max recognized words kept in NLP context.")
    parser.add_argument(
        "--nlp-cooldown-seconds",
        type=float,
        default=1.1,
        help="Cooldown to prevent repeatedly adding the same stable label as a new word.",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _open_cap(camera_index: int, backend: Optional[int]) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(camera_index)
    return cv2.VideoCapture(camera_index, backend)


def open_camera(camera_index: int, backend_choice: str) -> Tuple[cv2.VideoCapture, str]:
    if backend_choice == "auto":
        candidates: Tuple[Tuple[str, Optional[int]], ...] = (
            ("dshow", cv2.CAP_DSHOW),
            ("msmf", cv2.CAP_MSMF),
            ("default", None),
        )
    elif backend_choice == "default":
        candidates = (("default", None),)
    elif backend_choice == "msmf":
        candidates = (("msmf", cv2.CAP_MSMF),)
    else:
        candidates = (("dshow", cv2.CAP_DSHOW),)

    for backend_name, backend_code in candidates:
        cap = _open_cap(camera_index, backend_code)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        for _ in range(10):
            ok, _ = cap.read()
            if ok:
                break
        if ok:
            return cap, backend_name
        cap.release()

    attempted = ", ".join(name for name, _ in candidates)
    raise RuntimeError(
        f"Could not open camera index {camera_index} with backends: {attempted}. "
        "Close other camera apps (Zoom/Teams/Browser) and retry."
    )


def motion_score(prev_gray: Optional[np.ndarray], frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
    gray = cv2.cvtColor(cv2.resize(frame_bgr, (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 0.0, gray
    diff = cv2.absdiff(gray, prev_gray)
    score = float(np.mean(diff) / 255.0)
    return score, gray


def parse_label_csv(raw: str) -> List[str]:
    labels = [x.strip() for x in raw.split(",")]
    labels = [x for x in labels if x]
    return labels


def _safe_text(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def draw_panel(
    frame: np.ndarray,
    prediction_text: str,
    confidence: float,
    status: str,
    motion: float,
    seq_len: int,
    seq_target: int,
    top_items: List[Tuple[str, float]],
    words_text: str,
    sentence_text: str,
    input_mode: str,
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    panel_x0, panel_y0 = 12, 12
    panel_x1, panel_y1 = 730, 320
    cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.58, out, 0.42, 0.0, out)

    cv2.putText(out, f"Input: {input_mode}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 255), 1)
    cv2.putText(out, f"Prediction: {prediction_text}", (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 240, 120), 2)
    cv2.putText(out, f"Confidence: {confidence:.2f}", (24, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (65, 210, 255), 2)
    cv2.putText(out, f"Status: {status}", (24, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1)
    cv2.putText(out, f"Motion: {motion:.4f}", (24, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)

    bar_x0, bar_y0, bar_w, bar_h = 24, 160, 220, 14
    warmup = min(1.0, float(seq_len) / max(1.0, float(seq_target)))
    cv2.rectangle(out, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (90, 90, 90), 1)
    cv2.rectangle(out, (bar_x0 + 1, bar_y0 + 1), (bar_x0 + int((bar_w - 2) * warmup), bar_y0 + bar_h - 1), (80, 190, 255), -1)
    cv2.putText(
        out,
        f"Warmup: {seq_len}/{seq_target}",
        (bar_x0 + bar_w + 12, bar_y0 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
    )

    y = 196
    cv2.putText(out, "Top Predictions", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1)
    y += 18
    for label, prob in top_items:
        cv2.putText(out, f"{label:>12s}  {prob:.2f}", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (215, 215, 215), 1)
        px0, py0, pw, ph = 240, y - 11, 185, 10
        cv2.rectangle(out, (px0, py0), (px0 + pw, py0 + ph), (85, 85, 85), 1)
        fill = int(max(0.0, min(1.0, prob)) * (pw - 2))
        cv2.rectangle(out, (px0 + 1, py0 + 1), (px0 + 1 + fill, py0 + ph - 1), (120, 230, 140), -1)
        y += 22

    words_line = _safe_text(words_text if words_text else "(none)", max_len=38)
    sent_line = _safe_text(sentence_text if sentence_text else "(none)", max_len=38)
    cv2.putText(out, f"Words: {words_line}", (450, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1)
    cv2.putText(out, f"Sentence: {sent_line}", (450, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (120, 235, 150), 1)

    cv2.putText(
        out,
        "Press q/Esc to quit, c to clear words",
        (20, out.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model, idx_to_label, input_type = load_model_from_checkpoint(args.checkpoint, device, input_type_arg=args.input_type)
    label_to_idx = {name: idx for idx, name in idx_to_label.items()}
    nosign_idx = label_to_idx.get(args.nosign_label)
    require_hands_for_sign = not args.allow_zero_hands

    rgb_sequence = deque(maxlen=args.sequence_length) if input_type == "rgb" else None
    landmark_sequence = deque(maxlen=args.sequence_length) if input_type == "landmarks" else None
    landmark_extractor = TwoHandLandmarkExtractor() if input_type == "landmarks" else None

    proba_buffer = deque(maxlen=max(1, args.smoothing_window))
    pred_history = deque(maxlen=max(1, args.stable_window))
    motion_history = deque(maxlen=max(1, args.motion_window))
    idle_prior = np.ones(len(idx_to_label), dtype=np.float32) / max(1, len(idx_to_label))
    strict_labels = parse_label_csv(args.strict_labels)
    strict_idx = {label_to_idx[name] for name in strict_labels if name in label_to_idx}

    sentence_builder = OnlineSentenceBuilder(
        max_words=args.nlp_max_words,
        cooldown_seconds=args.nlp_cooldown_seconds,
    )

    cap, backend_name = open_camera(args.camera_index, args.camera_backend)
    print(f"Opened camera index {args.camera_index} using backend '{backend_name}'.")
    print(f"Input type: {input_type}")
    print(f"Loaded classes: {', '.join([idx_to_label[i] for i in sorted(idx_to_label.keys())])}")
    if nosign_idx is not None:
        print(f"No-sign class enabled: label='{args.nosign_label}', threshold={args.nosign_threshold:.2f}")
    else:
        print(f"No-sign class label '{args.nosign_label}' not found in checkpoint; suppression disabled.")
    if strict_idx:
        names = [idx_to_label[i] for i in sorted(strict_idx)]
        print(f"Strict labels enabled: {', '.join(names)}")
    else:
        print("Strict labels enabled: (none)")

    frame_idx = 0
    last_prediction: Optional[str] = "..."
    last_conf = 0.0
    last_status = "warming_up"
    last_top_items: List[Tuple[str, float]] = []
    read_failures = 0
    idle_frames = 0
    prev_gray: Optional[np.ndarray] = None
    hands_detected = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= 20:
                    print("Camera stream read failed repeatedly. Exiting.")
                    break
                continue
            read_failures = 0

            frame = cv2.flip(frame, 1)
            motion, prev_gray = motion_score(prev_gray, frame)
            motion_history.append(motion)
            motion_avg = float(np.mean(motion_history)) if motion_history else 0.0

            if input_type == "landmarks":
                if landmark_sequence is None or landmark_extractor is None:
                    raise RuntimeError("Landmark pipeline not initialized.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lms = landmark_extractor.extract(rgb)
                hands_detected = count_detected_hands(lms)
                landmark_sequence.append(lms)
            else:
                if rgb_sequence is None:
                    raise RuntimeError("RGB pipeline not initialized.")
                rgb_sequence.append(preprocess_frame_rgb(frame, args.image_size))
                hands_detected = 0
            frame_idx += 1

            current_len = len(landmark_sequence) if input_type == "landmarks" else len(rgb_sequence)
            ready = current_len == args.sequence_length
            classify_tick = frame_idx % max(1, args.predict_every) == 0

            if not ready:
                last_status = "warming_up"
            else:
                if classify_tick:
                    if input_type == "landmarks" and require_hands_for_sign and hands_detected <= 0:
                        pred_history.append("__uncertain__")
                        last_prediction = "..."
                        last_conf = 0.0
                        last_status = "no_hands"
                        last_top_items = []
                    else:
                        raw_probs = forward_probs(
                            model=model,
                            input_type=input_type,
                            rgb_sequence=rgb_sequence,
                            landmark_sequence=landmark_sequence,
                            seq_target=args.sequence_length,
                            device=device,
                            temperature=args.temperature,
                        )

                        if motion_avg < args.min_motion:
                            idle_frames += 1
                            alpha = float(max(0.0, min(1.0, args.idle_prior_alpha)))
                            idle_prior = alpha * idle_prior + (1.0 - alpha) * raw_probs
                            idle_prior = np.clip(idle_prior, 1e-6, None)
                            idle_prior = idle_prior / np.sum(idle_prior)
                            last_status = "idle_no_motion"
                            if idle_frames >= max(1, args.idle_reset_frames):
                                proba_buffer.clear()
                                pred_history.clear()
                                last_prediction = "..."
                                last_conf = 0.0
                                last_top_items = []
                        else:
                            idle_frames = 0
                            proba_buffer.append(raw_probs)
                            smoothed = np.mean(np.stack(proba_buffer, axis=0), axis=0)

                            debias = float(max(0.0, args.debias_strength))
                            if debias > 0.0:
                                denom = np.power(np.clip(idle_prior, 1e-4, 1.0), debias)
                                adjusted = smoothed / denom
                                adjusted = adjusted / max(1e-8, float(np.sum(adjusted)))
                            else:
                                adjusted = smoothed

                            tk = topk_probs(adjusted, max(3, args.top_k))
                            top_idx, top_conf = tk[0]
                            second_conf = tk[1][1] if len(tk) > 1 else 0.0
                            margin = top_conf - second_conf
                            nosign_prob = float(adjusted[nosign_idx]) if nosign_idx is not None else 0.0
                            suppress_for_nosign = nosign_idx is not None and nosign_prob >= float(args.nosign_threshold)
                            suppress_for_hands = (
                                input_type == "landmarks"
                                and require_hands_for_sign
                                and hands_detected <= 0
                            )

                            req_conf = args.strict_confidence if top_idx in strict_idx else args.min_confidence
                            req_margin = args.strict_margin if top_idx in strict_idx else args.min_margin
                            is_candidate_confident = top_conf >= req_conf and margin >= req_margin
                            chosen_idx = top_idx
                            chosen_conf = top_conf

                            # If strict label fails strict thresholds, try backup non-strict candidate.
                            if not is_candidate_confident and top_idx in strict_idx:
                                for alt_rank in range(1, len(tk)):
                                    alt_idx, alt_conf = tk[alt_rank]
                                    alt_second = tk[alt_rank + 1][1] if alt_rank + 1 < len(tk) else 0.0
                                    alt_margin = alt_conf - alt_second
                                    if alt_conf >= args.min_confidence and alt_margin >= args.min_margin:
                                        chosen_idx = alt_idx
                                        chosen_conf = alt_conf
                                        is_candidate_confident = True
                                        break

                            candidate = idx_to_label.get(chosen_idx, f"class_{chosen_idx}")
                            if nosign_idx is not None and chosen_idx == nosign_idx:
                                is_candidate_confident = False
                                last_status = "no_sign"
                            if suppress_for_nosign:
                                is_candidate_confident = False
                                last_status = "no_sign"
                            if suppress_for_hands:
                                is_candidate_confident = False
                                last_status = "no_hands"

                            pred_history.append(candidate if is_candidate_confident else "__uncertain__")
                            votes = Counter(pred_history)
                            if is_candidate_confident and votes[candidate] >= max(1, args.stable_votes):
                                last_prediction = candidate
                                last_conf = chosen_conf
                                last_status = "stable"
                                sentence_builder.add_label(candidate, ts=time.time())
                            elif is_candidate_confident:
                                last_status = "stabilizing"
                            else:
                                last_prediction = "..."
                                last_conf = chosen_conf
                                if last_status not in {"no_sign", "no_hands"}:
                                    last_status = "uncertain"

                            last_top_items = [(idx_to_label.get(i, f"class_{i}"), p) for i, p in tk[: args.top_k]]
                else:
                    if last_status not in {"stable", "uncertain"}:
                        last_status = "tracking"

            words_text = sentence_builder.words_text()
            sentence_text = sentence_builder.sentence()
            display = draw_panel(
                frame=frame,
                prediction_text=last_prediction or "...",
                confidence=last_conf,
                status=last_status,
                motion=motion_avg,
                seq_len=current_len,
                seq_target=args.sequence_length,
                top_items=last_top_items,
                words_text=words_text,
                sentence_text=sentence_text,
                input_mode=input_type,
            )
            cv2.imshow("ASL Live Recognition", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                sentence_builder.clear()
    finally:
        if landmark_extractor is not None:
            landmark_extractor.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
