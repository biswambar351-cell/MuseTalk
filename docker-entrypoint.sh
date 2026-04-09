#!/bin/bash
set -euo pipefail

cd /app

if [ "${DOWNLOAD_WEIGHTS:-0}" = "1" ] && [ ! -f "/app/models/musetalkV15/unet.pth" ]; then
  echo "Downloading MuseTalk weights..."
  bash /app/download_weights.sh
fi

exec python3 app.py \
  --ip "${GRADIO_SERVER_NAME:-0.0.0.0}" \
  --port "${PORT:-7860}" \
  $(if [ "${USE_FLOAT16:-1}" = "1" ]; then echo "--use_float16"; fi)
