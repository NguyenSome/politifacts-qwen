import argparse
import gc
import json
import logging
import logging.config
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pydantic import AnyUrl, BaseModel
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    action="ignore", message="The following generation flags are not valid"
)


class MemUsageCallback(TrainerCallback):
    def __init__(
        self, empty_cache_every_n_steps: int = 50, log_every_n_steps: int = 50
    ):
        """
        Args:
            empty_cache_every_n_steps: Clear CUDA cache every N steps (0 = never)
            log_every_n_steps: Log memory usage every N steps (1 = every step)
        """
        self.empty_cache_every_n_steps = empty_cache_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0

    def _log_mem(self, where: str):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"[{where}] [GPU {torch.cuda.current_device()}] "
            f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB",
            flush=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1

        should_log = self.step_count % self.log_every_n_steps == 0
        should_clear_cache = (
            self.empty_cache_every_n_steps > 0
            and self.step_count % self.empty_cache_every_n_steps == 0
        )

        # Log memory every N steps
        if should_log:
            self._log_mem("train_step_end")

        # Only run GC when we're doing memory operations to reduce overhead
        if should_log or should_clear_cache:
            gc.collect()

        # Clear cache every N steps
        if should_clear_cache:
            torch.cuda.empty_cache()
            print(
                f"[train_step_end] Cleared CUDA cache (step {self.step_count})",
                flush=True,
            )

    def on_evaluate(self, args, state, control, **kwargs):
        self._log_mem("on_evaluate_start")
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


LOGGER = logging.getLogger("my_app")

LABELS = [
    "pants-fire",
    "false",
    "mostly-false",
    "half-true",
    "mostly-true",
    "true",
]

bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
fp16_ok = torch.cuda.is_available()
dtype = torch.bfloat16 if bf16_ok else (torch.float16 if fp16_ok else torch.float32)


# Playing with pydantic
class MLflowSettings(BaseModel):
    enable: bool = False
    tracking_uri: AnyUrl | str | None = None
    exp_name: str = "default"
    run_name: str | None = None
    artifact_path: str = "artifacts"

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"finetune-{ts}"


def setup_logging() -> None:
    config_file = Path("configs/config.json")
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _read_config(path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {p}: {e}") from e

    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("lora", {})
    cfg.setdefault("mlflow", {})

    return cfg


def load_model(cfg: dict[str, Any]):
    """
    Parameters for model, tokenizer, and QLoRa training.
    Only a few parameters are added to configuration setting to spec as needed.
    """
    LOGGER.info("Loading base model %s", cfg["model"].get("model_name", "<missing>"))

    base_model = cfg["model"].get("model_name")

    # Qlora configuration - paper ref
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    base.config.use_cache = False  # Not needed during training
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(base, lora_cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def build_tokenize_fn(tokenizer, prompt: str, max_length: int):
    """
    Tokenize per-example, truncating the prompt to leave room for the target.
    """

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        statements: list[str] = [str(s).strip() for s in batch["statement"]]
        verdicts: list[str] = [str(v).strip() for v in batch["verdict"]]

        input_ids_all: list[list[int]] = []
        label_all: list[list[int]] = []

        orig_side = tokenizer.truncation_side
        tokenizer.truncation_side = "right"

        for s, v in zip(statements, verdicts, strict=False):
            text = prompt.format(statement=s)
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
            label_all.append(lbl)

        tokenizer.truncation_side = orig_side

        return {
            "input_ids": input_ids_all,
            "labels": label_all,
            "verdict": verdicts,
            "statement": statements,
        }

    return tokenize_batch


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def data_processing(cfgs, tokenizer, split_name: str = "training"):
    """
    Load JSON dataset, tokenize it, drop raw text columns, and return tokenized DatasetDict
    """

    data_path = cfgs["data"][split_name]
    ds = load_dataset(
        "json",
        data_files=data_path,
        keep_in_memory=False,
    )

    max_length = int(cfgs["data"].get("max_length", 1024))
    tokenize_fn = build_tokenize_fn(
        tokenizer,
        cfgs["prompt"]["template"],
        max_length,
    )

    if split_name == "training":
        split = ds["train"].train_test_split(test_size=0.15, seed=cfgs.get("seed", 7))
        dataset = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            }
        )
        cols_to_drop = dataset["train"].column_names

        return dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=32,
            num_proc=1,
            remove_columns=cols_to_drop,
            desc="Tokenizing (train/val)",
        )
    else:
        cols_to_drop = ds["train"].column_names

        tokenized = ds["train"].map(
            tokenize_fn,
            batched=True,
            batch_size=32,
            num_proc=1,
            remove_columns=cols_to_drop,
            desc=f"Tokenizing ({split_name})",
        )

        return DatasetDict({split_name: tokenized})


def decode_prediction_pairs(tokenizer, preds, labels) -> list[tuple[str, str]]:
    """Decode predicted/label token sequences into human-readable verdict strings."""

    if isinstance(preds, tuple):
        preds = preds[0]

    pred_array = np.array(preds)
    if pred_array.ndim == 3:
        pred_ids = pred_array.argmax(-1)
    else:
        pred_ids = pred_array

    label_ids = np.array(labels)
    pred_ids = np.array(pred_ids)

    decoded: list[tuple[str, str]] = []

    for i in range(label_ids.shape[0]):
        label_row = label_ids[i]
        pred_row = pred_ids[i]

        shifted_labels = label_row[1:]
        shifted_preds = pred_row[:-1]

        mask = shifted_labels != -100
        if not np.any(mask):
            continue

        true_ids = shifted_labels[mask]
        pred_span_ids = shifted_preds[mask]
        true_text = tokenizer.decode(true_ids, skip_special_tokens=True).strip().lower()
        pred_text = (
            tokenizer.decode(pred_span_ids, skip_special_tokens=True).strip().lower()
        )

        decoded.append((true_text, pred_text))

    return decoded


def build_metrics_computer(tokenizer):
    """
    Exact-string accuracy for the target only. Convert from logits -> ids when needed and compare on the label positions (mask != -100).
    """

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded = decode_prediction_pairs(tokenizer, preds, labels)
        y_true = [true for true, _ in decoded]
        y_pred = [pred for _, pred in decoded]

        accuracy = (
            float((np.array(y_pred) == np.array(y_true)).mean()) if y_true else 0.0
        )

        valid_idx = [i for i, p in enumerate(y_pred) if p in LABELS]
        if valid_idx:
            vt = [y_true[i] for i in valid_idx]
            vp = [y_pred[i] for i in valid_idx]

            macro_f1 = f1_score(vt, vp, labels=LABELS, average="macro", zero_division=0)
            kappa = cohen_kappa_score(vt, vp, labels=LABELS)
            recognized_ratio = len(valid_idx) / len(y_true)

            prec, rec, f1, supp = precision_recall_fscore_support(
                vt, vp, labels=LABELS, zero_division=0
            )

            metrics = {
                "accuracy": accuracy,
                "macro_f1": float(macro_f1),
                "cohen_kappa": float(kappa),
                "recognized_ratio": float(recognized_ratio),
            }

            for i, lab in enumerate(LABELS):
                metrics[f"per_class/precision/{lab}"] = float(prec[i])
                metrics[f"per_class/recall/{lab}"] = float(rec[i])
                metrics[f"per_class/f1/{lab}"] = float(f1[i])
                metrics[f"per_class/support/{lab}"] = float(supp[i])

        else:
            metrics = {
                "accuracy": accuracy,
                "macro_f1": 0.0,
                "cohen_kappa": 0.0,
                "recognized_ratio": 0.0,
            }

        return metrics

    return compute_metrics


def write_json(metrics: dict[str, float], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    LOGGER.info("Persisted metrics to %s", path)


def write_prediction_pairs(pairs: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for actual, predicted in pairs:
            fh.write(json.dumps({"actual": actual, "prediction": predicted}) + "\n")
    LOGGER.info("Persisted %d prediction pairs to %s", len(pairs), path)


def _manual_training_params() -> dict[str, Any]:
    """Tweak core training params."""

    return {
        "output_dir": Path("results/ar-qwen").resolve(),
        "logging_steps": 100,
        "saving_steps": 100,
        "eval_steps": 100,
    }


def train(model, tokenizer, cfgs: dict[str, Any], training_runtime: dict[str, Any]):
    mlflow_cfg = MLflowSettings(**cfgs["mlflow"])

    dataset = data_processing(cfgs, tokenizer, "training")
    compute_metrics = build_metrics_computer(tokenizer)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    out_dir = training_runtime["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt = get_last_checkpoint(str(out_dir)) if os.path.isdir(out_dir) else None
    LOGGER.info("Resuming from checkpoint: %s", last_ckpt)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="no",
        # eval_steps=training_runtime["eval_steps"],
        eval_accumulation_steps=4,
        save_strategy="steps",
        save_steps=training_runtime["saving_steps"],
        save_total_limit=2,
        logging_steps=training_runtime["logging_steps"],
        tf32=True,
        bf16=bf16_ok,
        fp16=(fp16_ok and not bf16_ok),
        max_grad_norm=0.3,
        dataloader_num_workers=0,
        optim="adamw_bnb_8bit",
        save_safetensors=True,
        report_to=["mlflow"] if mlflow_cfg.enable else [],
        run_name=mlflow_cfg.resolved_run_name(),
        remove_unused_columns=True,
        seed=cfgs["seed"],
        load_best_model_at_end=False,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=args,
        callbacks=[
            MemUsageCallback(),
            # EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    LOGGER.info(
        "Starting training run (epochs=%s, lr=%s)",
        args.num_train_epochs,
        args.learning_rate,
    )

    numeric_metrics: dict[str, float] = {}

    try:
        if mlflow_cfg.enable:
            tracking_uri = mlflow_cfg.tracking_uri or str(
                (Path.cwd() / "mlruns").resolve()
            )
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(mlflow_cfg.exp_name)
            LOGGER.info(
                "MLflow tracking enabled (experiment=%s, uri=%s)",
                mlflow_cfg.exp_name,
                tracking_uri,
            )
            with mlflow.start_run(run_name=mlflow_cfg.resolved_run_name()):
                LOGGER.info("Starting training")

                # Only log parameters if not resuming from checkpoint
                # (to avoid "Changing param values is not allowed" error)
                if last_ckpt is None:
                    try:
                        mlflow.log_params(
                            {
                                "model_name": cfgs["model"].get("model_name", ""),
                                "num_train_epochs": args.num_train_epochs,
                                "train_bs_per_device": args.per_device_train_batch_size,
                                "eval_bs_per_device": args.per_device_eval_batch_size,
                                "grad_accum": args.gradient_accumulation_steps,
                                "lr": args.learning_rate,
                                "weight_decay": args.weight_decay,
                                "scheduler": str(args.lr_scheduler_type),
                                "warmup_ratio": args.warmup_ratio,
                                "seed": int(cfgs.get("seed")),
                                "max_length": int(cfgs["data"].get("max_length", 1024)),
                            }
                        )
                    except Exception as e:
                        # If params already exist (e.g., run was reused), log a warning
                        LOGGER.warning(
                            "Could not log parameters (may already exist): %s", e
                        )
                else:
                    LOGGER.info("Skipping parameter logging (resuming from checkpoint)")

                train_output = trainer.train(resume_from_checkpoint=last_ckpt)
                train_metrics = train_output.metrics or {}
                eval_metrics = trainer.evaluate()  # eval_loss + compute_metrics()

                numeric_metrics = {
                    k: float(v)
                    for k, v in {**train_metrics, **eval_metrics}.items()
                    if isinstance(v, (int | float))
                }

                # safeguard in case of mlflow tracking errors
                metrics_path = out_dir / "training_metrics.json"
                write_json(numeric_metrics, metrics_path)
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

                prediction_output = trainer.predict(dataset["validation"])
                decoded_pairs = decode_prediction_pairs(
                    tokenizer,
                    prediction_output.predictions,
                    prediction_output.label_ids,
                )
                preds_path = out_dir / "validation_predictions.jsonl"
                write_prediction_pairs(decoded_pairs, preds_path)

                saved_model_dir = out_dir / "final_model"
                trainer.save_model(str(saved_model_dir))
                mlflow.log_artifacts(str(saved_model_dir), artifact_path="model")

        else:
            LOGGER.info("Starting training without MLflow tracking...")
            train_output = trainer.train(resume_from_checkpoint=last_ckpt)
            train_metrics = train_output.metrics or {}
            eval_metrics = trainer.evaluate()  # eval_loss + compute_metrics()

            numeric_metrics = {
                k: float(v)
                for k, v in {**train_metrics, **eval_metrics}.items()
                if isinstance(v, (int | float))
            }

            write_json(numeric_metrics, out_dir / "training_metrics.json")

            saved_model_dir = out_dir / "final_model"
            trainer.save_model(str(saved_model_dir))

    except Exception as e:
        LOGGER.error(f"Training failed: {e}")
        raise

    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    setup_logging()
    LOGGER.info("Logging setup complete")
    CONFIG_DIR = Path(args.config).resolve()
    cfgs = _read_config(CONFIG_DIR)
    model, tokenizer = load_model(cfgs)
    training_runtime = _manual_training_params()
    train(model, tokenizer, cfgs, training_runtime)


if __name__ == "__main__":
    main()
