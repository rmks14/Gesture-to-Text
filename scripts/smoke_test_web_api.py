from __future__ import annotations

import argparse
import base64
from typing import Dict

import cv2
import numpy as np
import requests


def post_json(url: str, payload: Dict) -> Dict:
    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def get_json(url: str) -> Dict:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def make_blank_data_url(width: int = 640, height: int = 360) -> str:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode test frame.")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for web_app API.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    health = get_json(f"{base}/api/health")
    start = post_json(f"{base}/api/start", {})
    data_url = make_blank_data_url()
    pred = post_json(f"{base}/api/predict", {"image": data_url})
    sentence = post_json(f"{base}/api/generate_sentence", {})
    cleared = post_json(f"{base}/api/clear", {})
    stopped = post_json(f"{base}/api/stop", {})

    assert health.get("ok"), f"health failed: {health}"
    assert start.get("ok"), f"start failed: {start}"
    assert pred.get("ok"), f"predict failed: {pred}"
    assert sentence.get("ok"), f"sentence failed: {sentence}"
    assert cleared.get("ok"), f"clear failed: {cleared}"
    assert stopped.get("ok"), f"stop failed: {stopped}"
    print("Smoke test passed.")
    print(
        f"health={health.get('status')} status={pred.get('status')} prediction={pred.get('prediction')} "
        f"hands={pred.get('hands_detected')} nosign_prob={pred.get('nosign_prob')}"
    )


if __name__ == "__main__":
    main()
