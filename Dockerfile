FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.txt

COPY src /app/src
COPY runs /app/runs
COPY README.md /app/README.md

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV CHECKPOINT_PATH=/app/runs/asl_landmarks_10cls_v2_light/best.pt \
    HOST=0.0.0.0 \
    PORT=8000 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=120

EXPOSE 8000

CMD ["sh", "-lc", "gunicorn --workers \"$GUNICORN_WORKERS\" --threads \"$GUNICORN_THREADS\" --timeout \"$GUNICORN_TIMEOUT\" --bind \"$HOST:$PORT\" src.wsgi:app"]
