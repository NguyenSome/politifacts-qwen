# aws-qwen

## Quickstart

```bash
# First time
source .venv/bin/activate
pre-commit install
pytest -q


python train.py --config configs/base.yaml --overrides configs/exp/lr-sweep-1.yaml