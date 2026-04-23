# End-to-End Sign Language Recognition (MediaPipe 2-Hand + BiLSTM + NLP Sentence Output)

This project is now an end-to-end **word-level sign recognition pipeline** with:

- automated clip expansion per word,
- **MediaPipe 2-hand landmarks** (`max_num_hands=2`),
- robust landmark preprocessing and augmentation,
- lightweight temporal-conv + BiLSTM + attention classifier,
- optional synthetic `no_sign` negatives (no personal data required),
- live inference with stabilization/debiasing,
- **NLP sentence generation** from recognized word labels.

Refactor/cleanup status:

- removed temporary source dumps (`tmp_*` directories),
- removed Python cache artifacts (`__pycache__`),
- centralized shared inference utilities in `src/inference_common.py`,
- added containerized runtime (`Dockerfile`, `docker-compose.yml`, `src/wsgi.py`) with `gunicorn` + API smoke-test script.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1b) Project Layout (Refactored)

- `src/inference_common.py`: shared checkpoint loading + preprocessing + forward helpers
- `src/train.py`: improved lightweight training pipeline (`temporal-conv`, focal loss, synthetic `no_sign`)
- `src/live_infer.py`: desktop camera inference
- `src/web_app.py`: browser-based web UI backend
- `src/web/templates/index.html`: web UI frontend
- `scripts/smoke_test_web_api.py`: API smoke test
- `Dockerfile` / `docker-compose.yml`: containerized deployment

## 2) Expand Word Clips (Add More Clips Per Word)

Top up each label to a target clip count from MS-ASL and WLASL:

```bash
python -m src.expand_word_clips ^
  --raw-dir data/raw ^
  --labels hello,how,no,ok,thank_you,yes,you,i,fine ^
  --target-per-label 180 ^
  --msasl-info-tar data/msasl/msasl.tar ^
  --wlasl-index-file tmp_wlasl/start_kit/WLASL_v0.3.json ^
  --max-variants-per-item 3 ^
  --jitter-seconds 0.30 ^
  --min-frames 12
```

## 3) Prepare Landmark Dataset (2-Hand MediaPipe)

```bash
python -m src.prepare_dataset ^
  --raw-dir data/raw ^
  --output-dir data/processed_landmarks ^
  --feature-type landmarks ^
  --sequence-length 32 ^
  --frame-step 1 ^
  --augment-copies 1 ^
  --min-hand-presence 0.20 ^
  --min-landmark-energy 0.0008 ^
  --landmark-smooth-alpha 0.65
```

Outputs:

- `data/processed_landmarks/metadata.csv`
- `data/processed_landmarks/label_map.json`
- `data/processed_landmarks/sequences/{train,val,test}/.../*.npz` (contains `features`)

## 4) Train (Improved Light Model + No-Sign Negatives)

```bash
python -m src.train ^
  --data-index data/processed_landmarks_9cls_v1/metadata.csv ^
  --label-map data/processed_landmarks_9cls_v1/label_map.json ^
  --output-dir runs/asl_landmarks_10cls_v2_light ^
  --input-type landmarks ^
  --epochs 65 ^
  --min-epochs 12 ^
  --early-stop-patience 12 ^
  --batch-size 32 ^
  --learning-rate 8e-4 ^
  --weight-decay 2e-4 ^
  --bidirectional ^
  --temporal-conv ^
  --temporal-conv-layers 2 ^
  --temporal-conv-kernel-size 5 ^
  --balanced-sampler ^
  --class-weighting ^
  --loss focal ^
  --focal-gamma 1.8 ^
  --model-select f1 ^
  --add-nosign-class ^
  --nosign-label no_sign ^
  --nosign-train-ratio 0.45 ^
  --nosign-val-ratio 0.25 ^
  --grad-clip 1.0 ^
  --lr-scheduler plateau
```

Saved checkpoints:

- `runs/asl_landmarks_10cls_v2_light/latest.pt`
- `runs/asl_landmarks_10cls_v2_light/best.pt`

Reference run (existing public dataset only, no personal data):

- best val macro-F1: `0.9622`
- test top-1 accuracy on lexical classes: `0.9464` (`265/280`)
- synthetic no-sign false-trigger rate at runtime thresholds (`0.55/0.12`, `no_sign>=0.50`): about `0.33%` (`2/600`)

## 5) Evaluate

```bash
python -m src.evaluate ^
  --data-index data/processed_landmarks_9cls_v1/metadata.csv ^
  --label-map data/processed_landmarks_9cls_v1/label_map.json ^
  --checkpoint runs/asl_landmarks_10cls_v2_light/best.pt ^
  --split test ^
  --input-type landmarks
```

Evaluation now reports:

- loss
- top-1 accuracy
- top-3 accuracy
- per-class accuracy

## 6) Live Inference + NLP Sentence Generation

```bash
python -m src.live_infer ^
  --checkpoint runs/asl_landmarks_10cls_v2_light/best.pt ^
  --input-type landmarks ^
  --sequence-length 32 ^
  --predict-every 2 ^
  --smoothing-window 8 ^
  --min-confidence 0.55 ^
  --min-margin 0.12 ^
  --nosign-label no_sign ^
  --nosign-threshold 0.50 ^
  --stable-window 5 ^
  --stable-votes 3 ^
  --nlp-max-words 10 ^
  --nlp-cooldown-seconds 1.1
```

Important: live landmark inference now applies the same per-sequence normalization used in
training/evaluation. This fixes the common collapse where predictions get stuck on one label
(for example repeatedly outputting `ok`).

Live window now shows:

- stabilized label prediction,
- rolling recognized words,
- NLP-generated sentence.

Controls:

- `q` / `Esc`: quit
- `c`: clear recognized word history

## 6b) Web UI (Start/Stop Camera + Guesses + Generate Sentence)

Run:

```bash
python -m src.web_app ^
  --checkpoint runs/asl_landmarks_10cls_v2_light/best.pt ^
  --input-type landmarks ^
  --min-stable-seconds 1.6 ^
  --nosign-label no_sign ^
  --nosign-threshold 0.50 ^
  --host 127.0.0.1 ^
  --port 8000
```

Then open `http://127.0.0.1:8000`.

Web controls:

- `Start Camera` / `Stop Camera`
- `Clear Guesses` (clears previous guessed labels + sentence context)
- right panel with live guessed labels and top predictions
- `Generate Sentence` button that runs the NLP sentence builder from recognized labels
- live **MediaPipe two-hand skeleton landmarks** overlay (both hands when detected)
- `no_sign` suppression to reduce false positive sign triggers during idle/background frames
- extra hold-time before accepting a label (`--min-stable-seconds`) to reduce premature guesses

## 6c) API Smoke Test

With app running locally:

```bash
python scripts/smoke_test_web_api.py --base-url http://127.0.0.1:8000
```

## 7) Docker

Container runtime uses `gunicorn` (`src.wsgi:app`) instead of Flask's dev server.

Build image:

```bash
docker build -t asl-sign-web:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 asl-sign-web:latest
```

Or with Compose:

```bash
docker compose up --build
```

If port `8000` is already in use on your host, override it:

```powershell
$env:HOST_PORT=18000; docker compose up --build
```

Then open `http://127.0.0.1:<HOST_PORT>` (default `8000`).
Health endpoint: `http://127.0.0.1:<HOST_PORT>/api/health`.

Container smoke test:

```bash
python scripts/smoke_test_web_api.py --base-url http://127.0.0.1:8000
```

## 8) One-Command Pipeline

```bash
python -m src.demo_pipeline ^
  --raw-dir data/raw ^
  --processed-dir data/processed_landmarks_demo ^
  --run-dir runs/asl_landmarks_demo ^
  --feature-type landmarks ^
  --epochs 20
```

## 9) Clean Previous Runs/Checkpoints

If you want a fresh workspace:

```powershell
Remove-Item -LiteralPath runs -Recurse -Force
Remove-Item -LiteralPath snapshots -Recurse -Force
```

## 10) Notes

- Landmark mode requires `mediapipe` in your environment.
- Older RGB datasets/checkpoints are still supported (`--feature-type rgb`, `--input-type rgb`).
- For best accuracy, keep clip quality consistent and remove noisy clips using `src.curate_raw_clips`.
