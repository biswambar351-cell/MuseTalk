#!/bin/bash
set -euo pipefail

cd /app

if [ "${DOWNLOAD_WEIGHTS:-0}" = "1" ] && [ ! -f "/app/models/musetalkV15/unet.pth" ]; then
  echo "Downloading MuseTalk weights..."
  bash /app/download_weights.sh
fi

exec python3 handler.py
