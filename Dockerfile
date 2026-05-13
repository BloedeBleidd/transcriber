FROM python:3.12.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_VERSION=26.0.1
ARG PRELOAD_MODELS=medium

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WHISPER_DOWNLOAD_ROOT=/opt/faster-whisper-cache \
    WHISPER_LOCAL_FILES_ONLY=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade "pip==${PIP_VERSION}" \
 && python -m pip install --no-cache-dir -r /app/requirements.txt \
 && python - <<'PY'
import importlib.metadata as m

expected = {
    "yt-dlp": "2026.3.17",
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.7.1",
    "huggingface-hub": "1.14.0",
    "tokenizers": "0.23.1",
    "onnxruntime": "1.25.1",
    "av": "17.0.1",
    "tqdm": "4.67.3",
}

for dist, ver in expected.items():
    got = m.version(dist)
    if got != ver:
        raise SystemExit(f"Version mismatch for {dist}: expected {ver}, got {got}")

print("Pinned Python dependencies verified.")
PY

RUN mkdir -p "${WHISPER_DOWNLOAD_ROOT}" \
 && PRELOAD_MODELS="${PRELOAD_MODELS}" python - <<'PY'
import os
from pathlib import Path
from faster_whisper.utils import download_model

raw = os.environ.get("PRELOAD_MODELS", "medium").strip()

# Only accept explicit comma-separated model names
models = [item.strip() for item in raw.split(",") if item.strip()]

if not models:
    raise SystemExit("PRELOAD_MODELS resolved to an empty model list")

print("Preloading faster-whisper models:")
for model_name in models:
    print(f"  - {model_name}")

download_root = os.environ["WHISPER_DOWNLOAD_ROOT"]
for model_name in models:
    model_path = download_model(
        model_name,
        cache_dir=download_root,
        local_files_only=False,
    )
    print(f"Downloaded: {model_name} -> {model_path}")

# Write manifest for fast checking in the wrapper script
manifest_path = Path(download_root) / "preloaded-models.txt"
with manifest_path.open("w") as f:
    for m in sorted(models):
        f.write(f"{m}\n")

print("All requested faster-whisper models are preloaded.")
PY

COPY app/transcriber.py /app/transcriber.py

ENTRYPOINT ["python", "/app/transcriber.py"]
