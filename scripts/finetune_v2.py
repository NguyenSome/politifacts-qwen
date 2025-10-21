"""
Fine-tuning script (v2) with fixes applied:

- Uses prompt template from config
- Correct per-example tokenization and truncation
- Pads labels via DataCollatorForSeq2Seq
- Correct TrainingArguments.evaluation_strategy
- Proper MLflow setup + metrics file output
- Saves adapter from the PEFT-wrapped model (no undefined var)
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import mlflow
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, cohen_kappa_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


LOGGER = logging.getLogger("my_app")

LABELS = [
    "pants-fire",
    "false",
    "mostly-false",
    "half-true",
    "mostly-true",
    "true",
]


# Auto-select dtype depending on hardware
bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
fp16_ok = torch.cuda.is_available()
dtype = torch.bfloat16 if bf16_ok else (torch.float16 if fp16_ok else torch.float32)


def setup_logging() -> None:
    config_file = Path("configs/config.json")
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _read_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {p}: {e}") from e

    return data


def _get_cfg(path: str | Path = "configs/base.yaml") -> Dict[str, Any]:
    cfg = _read_config(path)
    # Normalize/guard expected sections
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("prompt", {})
    cfg.setdefault("lora", {})
    cfg.setdefault("trainer", {})
    cfg.setdefault("mlflow", {})
    return cfg


def _get_prompt_template(cfg: Dict[str, Any]) -> str:
    default_tmpl = (
        "Classify the following statement with one label only, "
        "chosen from: pants-fire, false, mostly-false, half-true, mostly-true, true.\n"
        "Statement: {statement}\n"
        "Answer with only the label, nothing else:\n"
    )
    prompt_section = cfg.get("prompt", {}) or {}
    return prompt_section.get("template", default_tmpl)


def load_model(cfg: Dict[str, Any]):
    LOGGER.info("Loading base model %s", cfg["model"].get("model_name", "<missing>"))

    model_name = cfg["model"].get("model_name")
    if not model_name:
        raise ValueError("config[model][model_name] is required")

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA config (optional enable)
    lora_cfg = cfg.get("lora", {}) or {}
    if lora_cfg.get("enable", True):
        target_modules = lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        lora = LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base, lora)
    else:
        model = base

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def build_tokenize_fn(tokenizer, prompt_template: str, max_length: int):
    """
    Tokenize per-example, truncating the prompt to leave room for the target.
    Returns variable-length lists; collator will handle padding.
    """

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        statements: List[str] = [str(s).strip() for s in batch["statement"]]
        verdicts: List[str] = [str(v).strip() for v in batch["verdict"]]

        input_ids_all: List[List[int]] = []
        labels_all: List[List[int]] = []

        orig_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"

        for s, v in zip(statements, verdicts):
            text = prompt_template.format(statement=s)
            target = " " + v

            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            available = max(0, max_length - len(target_ids))
            prompt_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=available,
            )["input_ids"]

            ids = prompt_ids + target_ids
            lbl = ([-100] * len(prompt_ids)) + target_ids

            input_ids_all.append(ids)
            labels_all.append(lbl)

        tokenizer.truncation_side = orig_side

        return {
            "input_ids": input_ids_all,
            "labels": labels_all,
            "verdict": verdicts,
        }

    return tokenize_batch


def data_processing(cfgs: Dict[str, Any], tokenizer, prompt_template: str) -> DatasetDict:
    ds = load_dataset("json", data_files=cfgs["data"]["training"])
    split = ds["train"].train_test_split(test_size=0.15, seed=cfgs.get("seed", 42))
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    tokenize_fn = build_tokenize_fn(tokenizer, prompt_template, int(cfgs["data"].get("max_length", 256)))

    LOGGER.info(
        "Tokenizing dataset: %s train / %s validation",
        len(dataset["train"]),
        len(dataset["validation"]),
    )
    cols_to_drop = dataset["train"].column_names
    processed = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=cols_to_drop,
        desc="Tokenizing",
    )
    return processed


def build_metrics_computer(tokenizer):
    """
    Exact-string accuracy for the target only. Convert from logits -> ids when needed and compare
    on the label positions (mask != -100). Also compute macro F1 and Cohen's kappa over recognized
    labels.
    """

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if preds.ndim == 3:
            pred_ids = preds.argmax(-1)
        else:
            pred_ids = preds

        labels = np.array(labels)
        pred_ids = np.array(pred_ids)

        y_true: List[str] = []
        y_pred: List[str] = []

        for i in range(labels.shape[0]):
            mask = labels[i] != -100
            if not np.any(mask):
                continue
            true_ids = labels[i][mask]
            pred_span_ids = pred_ids[i][mask]

            true_text = tokenizer.decode(true_ids, skip_special_tokens=True).strip().lower()
            pred_text = tokenizer.decode(pred_span_ids, skip_special_tokens=True).strip().lower()

            y_true.append(true_text)
            y_pred.append(pred_text)

        accuracy = float((np.array(y_pred) == np.array(y_true)).mean()) if y_true else 0.0

        valid_idx = [i for i, p in enumerate(y_pred) if p in LABELS]
        if valid_idx:
            vt = [y_true[i] for i in valid_idx]
            vp = [y_pred[i] for i in valid_idx]
            macro_f1 = f1_score(vt, vp, labels=sorted(LABELS), average="macro", zero_division=0)
            kappa = cohen_kappa_score(vt, vp, labels=sorted(LABELS))
            recognized_ratio = len(valid_idx) / len(y_true)
        else:
            macro_f1 = 0.0
            kappa = 0.0
            recognized_ratio = 0.0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "cohen_kappa": kappa,
            "recognized_ratio": recognized_ratio,
        }

    return compute_metrics


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Persisted metrics to %s", path)


def _resolve_training_args(cfgs: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    tcfg = cfgs.get("trainer", {}) or {}

    # Map YAML's eval_strategy to HF's evaluation_strategy
    evaluation_strategy = tcfg.get("evaluation_strategy", tcfg.get("eval_strategy", "epoch"))

    args: Dict[str, Any] = dict(
        output_dir=str(out_dir),
        num_train_epochs=int(tcfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 2)),
        learning_rate=float(tcfg.get("learning_rate", 5e-5)),
        evaluation_strategy=evaluation_strategy,
        save_strategy=str(tcfg.get("save_strategy", "epoch")),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        bf16=bf16_ok,
        fp16=(fp16_ok and not bf16_ok),
        report_to=["mlflow"] if (cfgs.get("mlflow", {}).get("enable", False)) else [],
        run_name=_resolved_run_name(cfgs.get("mlflow", {})),
        remove_unused_columns=bool(tcfg.get("remove_unused_columns", True)),
        seed=int(cfgs.get("seed", 42)),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )
    return args


def _resolved_run_name(mlcfg: Dict[str, Any]) -> str:
    rn = mlcfg.get("run_name")
    if rn:
        return str(rn)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"finetune-{ts}"


def _setup_mlflow_if_enabled(mlcfg: Dict[str, Any]) -> None:
    if not mlcfg.get("enable", False):
        return
    try:
        tracking_uri = mlcfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(str(tracking_uri))
        exp_name = mlcfg.get("exp_name") or "default"
        mlflow.set_experiment(exp_name)
        LOGGER.info("MLflow tracking enabled (experiment=%s)", exp_name)
    except Exception as e:
        LOGGER.warning("Failed to set up MLflow: %s", e)


def train(model, tokenizer, cfgs: Dict[str, Any]):
    prompt_tmpl = _get_prompt_template(cfgs)
    dataset = data_processing(cfgs, tokenizer, prompt_tmpl)
    compute_metrics = build_metrics_computer(tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)

    out_dir = Path(cfgs.get("trainer", {}).get("output_dir", "./ar-qwen-mini")).resolve()
    args_dict = _resolve_training_args(cfgs, out_dir)
    args = TrainingArguments(**args_dict)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    # Early stopping with default patience
    trainer.add_callback(EarlyStoppingCallback())

    numeric_metrics: Dict[str, float] = {}
    mlcfg = cfgs.get("mlflow", {}) or {}
    _setup_mlflow_if_enabled(mlcfg)

    try:
        if mlcfg.get("enable", False):
            with mlflow.start_run(run_name=_resolved_run_name(mlcfg)):
                LOGGER.info("Starting training with MLflow tracking...")
                trainer.train()
                metrics = trainer.evaluate()
                numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(numeric_metrics)
                write_json(out_dir / "metrics.json", numeric_metrics)
                # Best-effort artifact logging
                for path, target in [
                    ("config.effective.yaml", "config"),
                    ("results/metrics.json", "metrics"),
                    ("data/card.yaml", "data"),
                ]:
                    p = Path(path)
                    if p.exists():
                        try:
                            mlflow.log_artifact(str(p), artifact_path=target)
                            LOGGER.info("Logged artifact %s -> %s", p, target)
                        except Exception as e:
                            LOGGER.warning("Failed to log artifact %s: %s", p, e)
        else:
            LOGGER.info("Starting training without MLflow tracking...")
            trainer.train()
            metrics = trainer.evaluate()
            numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            write_json(out_dir / "metrics.json", numeric_metrics)

        # Save models
        model_dir = Path("models/v1/QWEN2.5").resolve()
        adapter_dir = Path("models/v1/qwen2.5-v1-adapter").resolve()
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        model.save_pretrained(str(adapter_dir))  # PEFT-wrapped model saves adapters
        LOGGER.info("Training complete and outputs saved")

    except Exception as e:
        LOGGER.error("Training failed: %s", e)
        raise


def main() -> None:
    setup_logging()
    LOGGER.info("Logging setup complete")
    cfgs = _get_cfg("configs/base.yaml")
    model, tokenizer = load_model(cfgs)
    train(model, tokenizer, cfgs)


if __name__ == "__main__":
    main()

