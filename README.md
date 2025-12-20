# Qwen2.5 Fact-Checking Classifier (PolitiFact) 🛰️

Pipeline for fine-tuning Qwen2.5 on the [PolitiFact](https://www.politifact.com/) fact-checking
dataset, with:

- Config-driven training (YAML)
- QLoRA fine-tuning on GPU-constrained hardware
- MLflow experiment tracking
- Dockerized training & evaluation
- AWS (EC2 + ECR) workflow

This repo is structured to showcase a workflow from dataset → training → evaluation → experiment tracking → packaged Docker
image that can run locally or in the cloud.

---

## Project Overview

- **Goal**: Classify political statements into discrete truthfulness labels  
  (e.g. `pants-fire`, `false`, `mostly-false`, `half-true`, `mostly-true`, `true`).
- **Model**: Qwen2.5-0.5B fine-tuned with QLoRA.
- **Stack**:
  - Python 3.11
  - PyTorch + Transformers
  - PEFT / QLoRA
  - MLflow (local tracking via `mlflow.db` or remote tracking URI)
  - Docker + NVIDIA CUDA 12.9
  - (Optional) AWS EC2 + ECR for training in the cloud

---

## Quickstart (Local)

**Prereqs**: Python 3.11, CUDA 12.x + NVIDIA GPU for training

```bash
# Install dependencies (recommended: uv)
uv sync

# Zero-shot baseline (fast sanity check)
uv run src/zero_shot_eval.py --config configs/test.yaml

# Fine-tune with QLoRA
uv run src/finetune.py --config configs/base.yaml
```

---

## Data

The `data/` directory contains JSONL files (one record per line). Key fields include:
`statement`, `verdict`, `statement_originator`, `statement_source`,
`statement_date`, and `factcheck_analysis_link`.

Use `small_train.json` or `micro_test.json` for quick experiments.

---

## Configuration

- `configs/base.yaml`: full training run.
- `configs/test.yaml`: short debug run with smaller settings.

---

## Training & Evaluation

```bash
make train_local     # QLoRA fine-tuning with configs/base.yaml
make demo_model      # Short fine-tune run with configs/test.yaml
make demo_base       # Zero-shot baseline with configs/test.yaml
```

---

## MLflow Tracking

Local tracking uses `mlflow.db` by default:

```bash
make show_mlflow
# or
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

---

## Docker

Build and run with GPU passthrough:

```bash
make build
make train     # runs src/finetune.py in the container
make eval      # runs src/zero_shot_eval.py in the container
make mlflow_ui # runs MLflow UI inside the container
make shell     # drop into container shell
```

---

## AWS (Optional)

AWS helpers assume `.env` variables:

```bash
make aws_spot
make ecr_login
make build_and_push
```

---

## Repository Structure

- `configs/` - Experiment configs
  - `base.yaml` - Base experiment configuration
  - `test.yaml` - Script testing configuration
- `data/` - Postprocessed PolitiFact data obtained from Kaggle
- `src/`
  - `entrypoint.sh` - Container entrypoint
  - `finetune.py` - Training entrypoint (QLoRA fine-tuning, MLflow logging)
  - `zero_shot_eval.py` - Baseline model testing (metrics, confusion matrix, etc.)
- `notebooks/` - (not committed) Exploration / EDA / debugging
- `scripts/` - (not committed) Exploration / debugging
- `models/` - (not committed) saved model checkpoints
- `results/` - (not committed) evaluation artifacts
- `logs/` - (not committed) log files
- `mlruns/` - (not committed) mlflow logs and artifacts
- `Dockerfile`
- `Makefile`
- `pyproject.toml`
- `requirements.txt`
- `spot-spec.json` - (not committed) EC2 spot instance config
- `uv.lock`
- `README.md`
