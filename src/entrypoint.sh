#!/bin/bash
set -euo pipefail

trap 'echo "[entrypoint] Received termination signal"; exit 143' SIGTERM SIGINT

CMD=${1:-finetune}
CONFIG=${CONFIG:-configs/base.yaml}

export GIT_PYTHON_REFRESH=quiet

echo "[entrypoint] Python version: $(python -V)"
echo "[entrypoint] NVIDIA visible devices: ${NVIDIA_VISIBLE_DEVICES:-none}"
if CUDA_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null); then
  echo "[entrypoint] CUDA device count: ${CUDA_COUNT}"
else
  echo "[entrypoint] CUDA device count: unavailable"
fi
echo "[entrypoint] Command: ${CMD} | Config: ${CONFIG}"

# Shift off the subcommand if present
if [[ $# -gt 0 ]]; then
  shift
fi

case "$CMD" in
  finetune)
    exec python -u src/finetune.py --config "$CONFIG" "$@"
    ;;
  eval)
    exec python -u src/zero_shot_eval.py --config "$CONFIG" "$@"
    ;;
  python|bash|sh)
    exec "$CMD" "$@"
    ;;
  *.py)
    exec python -u "$CMD" "$@"
    ;;
  *)
    echo "[entrypoint] Unknown command: $CMD"
    echo "[entrypoint] Usage:"
    echo "  finetune [--config path] [extra args]"
    echo "  eval [--config path] [extra args]"
    echo "  python|bash|sh ..."
    echo "  path/to/script.py [args]"
    exit 1
    ;;
esac
