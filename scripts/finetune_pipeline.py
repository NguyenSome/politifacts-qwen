"""A production-ready fine-tuning pipeline for Qwen models with MLflow integration.

This module refactors the ad-hoc prototype in ``finetune.py`` into a maintainable
training pipeline that can back a Streamlit and MLflow monitoring stack.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import cohen_kappa_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

import paths

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

CANONICAL_LABELS: Tuple[str, ...] = (
    "pants-fire",
    "false",
    "mostly-false",
    "half-true",
    "mostly-true",
    "true",
)


@dataclass
class MLflowSettings:
    """Configuration for optional MLflow tracking."""

    enable: bool = False
    tracking_uri: Optional[str] = None
    experiment_name: str = "qwen-finetune"
    run_name: Optional[str] = None
    artifact_path: str = "finetuned-model"

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return f"qwen-finetune-{timestamp}"


@dataclass
class TrainingConfig:
    """Strongly-typed configuration payload for the training pipeline."""

    model_name: str
    max_length: int
    data_path: Path
    train_filename: str = "train.json"
    validation_filename: Optional[str] = None
    validation_split: float = 0.15
    seed: int = 42
    num_epochs: int = 3
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    save_total_limit: int = 2
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    output_dir: Path = field(default_factory=lambda: paths.RESULTS_DIR / "finetune_runs")
    metrics_output: Path = field(default_factory=lambda: paths.RESULTS_DIR / "finetune_metrics.json")
    prompt_template: str = (
        "Classify the following statement with one label only, "
        "chosen from: pants-fire, false, mostly-false, half-true, mostly-true, true.\n\n"
        "Statement: {statement}\n"
        "Answer with only the label, nothing else:\n"
    )
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_bf16: Optional[bool] = None
    use_fp16: Optional[bool] = None
    early_stopping_patience: int = 2
    mlflow: MLflowSettings = field(default_factory=MLflowSettings)


def load_training_config(config_path: Path) -> TrainingConfig:
    """Load and validate configuration from a YAML file."""

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    required_keys = ("model_name", "max_length", "data_path")
    missing = [key for key in required_keys if key not in raw]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Configuration missing required fields: {missing_str}")

    mlflow_raw = raw.get("mlflow", {}) or {}
    mlflow_settings = MLflowSettings(
        enable=bool(mlflow_raw.get("enable", False)),
        tracking_uri=mlflow_raw.get("tracking_uri"),
        experiment_name=mlflow_raw.get("exp_name", "qwen-finetune"),
        run_name=mlflow_raw.get("run_name"),
        artifact_path=mlflow_raw.get("artifact_path", "finetuned-model"),
    )

    def resolve_path(value: Optional[str], default: Optional[Path] = None) -> Path:
        if value:
            path = Path(value)
            if not path.is_absolute():
                return paths.get_path(value)
            return path
        if default is None:
            raise ValueError("Path resolution failed and no default provided")
        return default

    lora_raw = raw.get("lora", {}) or {}

    # Determine metrics output path with backward compatibility:
    # Prefer top-level `metrics_output`, else fallback to `mlflow.results` if present,
    # else default to results/finetune_metrics.json
    metrics_out_cfg = raw.get("metrics_output")
    if not metrics_out_cfg:
        metrics_out_cfg = mlflow_raw.get("results")

    config = TrainingConfig(
        model_name=str(raw["model_name"]),
        max_length=int(raw["max_length"]),
        data_path=resolve_path(str(raw["data_path"])),
        train_filename=str(raw.get("train_file", "train.json")),
        validation_filename=raw.get("validation_file"),
        validation_split=float(raw.get("validation_split", 0.15)),
        seed=int(raw.get("seed", 42)),
        num_epochs=int(raw.get("num_epochs", 3)),
        train_batch_size=int(raw.get("train_batch_size", 2)),
        eval_batch_size=int(raw.get("eval_batch_size", 2)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
        learning_rate=float(raw.get("learning_rate", 5e-5)),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.0)),
        logging_steps=int(raw.get("logging_steps", 10)),
        save_total_limit=int(raw.get("save_total_limit", 2)),
        gradient_checkpointing=bool(raw.get("gradient_checkpointing", False)),
        max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
        output_dir=resolve_path(raw.get("output_dir"), paths.RESULTS_DIR / "finetune_runs"),
        metrics_output=resolve_path(metrics_out_cfg, paths.RESULTS_DIR / "finetune_metrics.json"),
        prompt_template=str(raw.get("prompt_template", TrainingConfig.prompt_template)),
        lora_r=int(lora_raw.get("r", raw.get("lora_r", TrainingConfig.lora_r))),
        lora_alpha=int(lora_raw.get("alpha", raw.get("lora_alpha", TrainingConfig.lora_alpha))),
        lora_dropout=float(lora_raw.get("dropout", raw.get("lora_dropout", TrainingConfig.lora_dropout))),
        target_modules=tuple(lora_raw.get("target_modules", raw.get("target_modules", TrainingConfig.target_modules))),
        use_bf16=raw.get("use_bf16"),
        use_fp16=raw.get("use_fp16"),
        early_stopping_patience=int(raw.get("early_stopping_patience", TrainingConfig.early_stopping_patience)),
        mlflow=mlflow_settings,
    )

    if not 0 < config.validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1 (exclusive)")

    return config


def set_seed(seed: int) -> None:
    """Deterministic training across popular frameworks."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def select_dtype(config: TrainingConfig) -> torch.dtype:
    """Resolve optimal torch dtype respecting explicit overrides."""

    if config.use_bf16 is True:
        return torch.bfloat16
    if config.use_fp16 is True:
        return torch.float16
    if config.use_bf16 is False and config.use_fp16 is False:
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(config: TrainingConfig, dtype: torch.dtype):
    """Load base model, apply LoRA, and prepare tokenizer."""

    LOGGER.info("Loading base model %s", config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=list(config.target_modules),
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def _normalize_label(label: str) -> str:
    """Lowercase normalization with whitespace and punctuation trimmed."""

    text = label.strip().lower()
    text = text.replace(" ", "-")
    return text


def build_tokenizer_fn(config: TrainingConfig, tokenizer) -> Any:
    """Create a closure that tokenizes single examples."""

    max_length = config.max_length

    def tokenize_example(example: Dict[str, Any]) -> Dict[str, Any]:
        statement = str(example["statement"]).strip()
        verdict = str(example["verdict"]).strip()

        prompt = config.prompt_template.format(statement=statement)
        target = " " + verdict

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

        if len(target_ids) >= max_length:
            target_ids = target_ids[: max_length - 1]

        max_prompt_tokens = max(0, max_length - len(target_ids))
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        target_len = min(len(target_ids), max_length - len(prompt_ids))
        target_ids = target_ids[:target_len]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids[:]

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        seq_len = max_length - pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "y_start": len(prompt_ids),
            "y_end": len(prompt_ids) + len(target_ids),
            "label_text": verdict,
        }

    return tokenize_example


def prepare_datasets(config: TrainingConfig, tokenizer) -> DatasetDict:
    """Load source data, split, and apply tokenization."""

    train_path = config.data_path / config.train_filename
    data_files: Dict[str, str] = {"train": str(train_path)}

    if config.validation_filename:
        val_path = config.data_path / config.validation_filename
        data_files["validation"] = str(val_path)

    dataset = load_dataset("json", data_files=data_files)

    if "validation" not in dataset:
        LOGGER.info("Creating validation split (%.2f)", config.validation_split)
        split_dataset = dataset["train"].train_test_split(
            test_size=config.validation_split,
            seed=config.seed,
        )
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })
    else:
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["validation"],
        })

    tokenize_fn = build_tokenizer_fn(config, tokenizer)

    LOGGER.info("Tokenising dataset: %s train / %s validation", len(dataset["train"]), len(dataset["validation"]))
    processed = dataset.map(
        tokenize_fn,
        remove_columns=list(dataset["train"].column_names),
        desc="Tokenising",
    )
    return processed


def compute_metrics_builder(eval_dataset, tokenizer):
    """Return a metric computation function bound to dataset and tokenizer."""

    def compute_metrics(predictions_and_labels: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        predictions, labels = predictions_and_labels

        if predictions.ndim == 3:
            pred_ids = predictions.argmax(-1)
        else:
            pred_ids = predictions

        total = len(pred_ids)
        correct = 0
        preds_normalized: List[str] = []
        trues_normalized: List[str] = []

        for idx in range(total):
            sample = eval_dataset[idx]
            y_start = int(sample["y_start"])
            y_end = int(sample["y_end"])
            true_text = _normalize_label(sample["label_text"])

            token_slice = pred_ids[idx, y_start:y_end]
            pred_text = tokenizer.decode(token_slice, skip_special_tokens=True).strip().lower()
            pred_label = _normalize_label(pred_text) if pred_text else ""

            if pred_label == true_text and pred_label in CANONICAL_LABELS:
                correct += 1

            preds_normalized.append(pred_label if pred_label in CANONICAL_LABELS else "unknown")
            trues_normalized.append(true_text)

        accuracy = correct / total if total else 0.0

        valid_indices = [i for i, pred in enumerate(preds_normalized) if pred in CANONICAL_LABELS]
        if valid_indices:
            valid_true = [trues_normalized[i] for i in valid_indices]
            valid_pred = [preds_normalized[i] for i in valid_indices]
            macro_f1 = f1_score(valid_true, valid_pred, labels=list(CANONICAL_LABELS), average="macro", zero_division=0)
            kappa = cohen_kappa_score(valid_true, valid_pred, labels=list(CANONICAL_LABELS))
        else:
            macro_f1 = 0.0
            kappa = 0.0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "cohen_kappa": kappa,
            "recognized_ratio": len(valid_indices) / total if total else 0.0,
        }

    return compute_metrics


def create_training_arguments(config: TrainingConfig, dtype: torch.dtype) -> TrainingArguments:
    """Instantiate TrainingArguments with sensible defaults."""

    bf16 = dtype == torch.bfloat16
    fp16 = dtype == torch.float16

    return TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=True,
        report_to=["mlflow"] if config.mlflow.enable else [],
        bf16=bf16,
        fp16=fp16,
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=config.max_grad_norm,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        run_name=config.mlflow.resolved_run_name() if config.mlflow.enable else None,
    )


def ensure_directory(path: Path) -> None:
    """Create directory tree if missing."""

    path.mkdir(parents=True, exist_ok=True)


def write_metrics_to_disk(metrics: Dict[str, float], path: Path) -> None:
    """Persist evaluation metrics in JSON for downstream dashboards."""

    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("Persisted metrics to %s", path)


def log_params_to_mlflow(config: TrainingConfig) -> None:
    """Log primary hyperparameters to MLflow."""

    params = {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
    }
    mlflow.log_params(params)


def run_training(config: TrainingConfig, config_path: Path) -> Dict[str, float]:
    """Execute the end-to-end training loop."""

    set_seed(config.seed)
    ensure_directory(config.output_dir)

    dtype = select_dtype(config)
    model, tokenizer = load_model_and_tokenizer(config, dtype)
    datasets = prepare_datasets(config, tokenizer)

    compute_metrics = compute_metrics_builder(datasets["validation"], tokenizer)
    training_args = create_training_arguments(config, dtype)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if config.early_stopping_patience > 0:
        early_stopping = EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        trainer.add_callback(early_stopping)

    numeric_metrics: Dict[str, float] = {}

    if config.mlflow.enable:
        if config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        run_name = config.mlflow.resolved_run_name()
        with mlflow.start_run(run_name=run_name):
            mlflow.log_artifact(str(config_path), artifact_path="config")
            log_params_to_mlflow(config)
            trainer.train()
            metrics = trainer.evaluate()
            numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(numeric_metrics)
            write_metrics_to_disk(numeric_metrics, config.metrics_output)
            # Log metrics JSON and final model directory as MLflow artifacts
            try:
                mlflow.log_artifact(str(config.metrics_output), artifact_path="metrics")
            except Exception as _:
                LOGGER.warning("Could not log metrics artifact to MLflow")
            try:
                mlflow.log_artifacts(str(config.output_dir), artifact_path=config.mlflow.artifact_path)
            except Exception as _:
                LOGGER.warning("Could not log model artifacts to MLflow")
    else:
        trainer.train()
        metrics = trainer.evaluate()
        numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        write_metrics_to_disk(numeric_metrics, config.metrics_output)

    trainer.save_model()
    tokenizer.save_pretrained(str(config.output_dir))

    return numeric_metrics


def parse_args() -> argparse.Namespace:
    """CLI entry point for the pipeline."""

    parser = argparse.ArgumentParser(description="Fine-tune Qwen models with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "eval.yaml"),
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_training_config(config_path)
    metrics = run_training(config, config_path)
    LOGGER.info("Final evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
