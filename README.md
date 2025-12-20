# Qwen2.5 Fact-Checking Classifier (Politifact) 🛰️

Pipeline for fine-tuning Qwen2.5 on the [Politifact] fact-checking dataset, with:

- Config-driven training (YAML)
- QLoRA fine-tuning on GPU-constrained hardware
- MLflow experiment tracking
- Dockerized training & evaluation
- AWS-ready (EC2 + ECR) workflow

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

## Repository Structure

├── configs/
│   ├── base.yaml         # Base experiment configuration
│   ├── test.yaml         # Script testing configuration
├── data/                 # Postprocessed Politifacts data obtained from Kaggle
├── src/
│   ├── finetune.py          # Training entrypoint (QLoRA fine-tuning, MLflow logging)
│   ├── zero_shot_eval.py    # Baseline model testing (metrics, confusion matrix, etc.)
├── notebooks/            # (not committed) Exploration / EDA / debugging 
├── scripts/              # (not committed) Exploration / debugging
├── models/               # (not committed) saved model checkpoints
├── results/              # (not committed) evaluation artifacts
├── logs/                 # (not committed) log files
├── mlruns/               # (not committed) mlflow logs and artifacts
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md