import argparse
import json, yaml
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import torch, numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, cohen_kappa_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from pydantic import BaseModel, AnyUrl


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", message="The following generation flags are not valid") 

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
dtype   = torch.bfloat16 if bf16_ok else (torch.float16 if fp16_ok else torch.float32)

MLFLOW_KEYMAP = {
    # eval
    "eval_loss":                 "val/loss",
    "eval_accuracy":             "val/accuracy",
    "eval_macro_f1":             "val/f1_macro",
    "eval_cohen_kappa":          "val/kappa",
    "eval_recognized_ratio":     "val/recognized_ratio",
    "eval_runtime":              "time/eval_runtime_sec",
    "eval_samples_per_second":   "time/eval_examples_per_sec",
    "eval_steps_per_second":     "time/eval_steps_per_sec",
    # train
    "train_runtime":             "time/train_runtime_sec",
    "train_samples_per_second":  "time/train_examples_per_sec",
    "train_steps_per_second":    "time/train_steps_per_sec",
}


class MLflowSettings(BaseModel):
    enable: bool = False
    tracking_uri: Optional[AnyUrl | str] = None
    exp_name: str = "default"
    run_name: Optional[str] = None
    artifact_path: str = "artifacts"

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"finetune-{ts}"
    
    
def _mlflow_log_metrics_renamed(metrics: Dict[str, float]) -> None:
    renamed = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        newk = MLFLOW_KEYMAP.get(k, k)
        renamed[newk] = float(v)
    if renamed:
        mlflow.log_metrics(renamed)


def setup_logging() -> None:
    config_file = Path("configs/config.json")
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _read_config(path: str | Path = "configs/base.yaml") -> Dict[str, Any]:
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
    cfg.setdefault("trainer", {})
    cfg.setdefault("mlflow", {})

    return cfg


def load_model(cfg: Dict[str, Any]):
    """
    Parameters for model, tokenizer, and LoRa based on RTX 3070 specs. 
    Only a few parameters are added to configuration setting to spec as needed.
    """
    LOGGER.info("Loading base model %s", cfg["model"].get("model_name", "<missing>"))
    
    base_model = cfg["model"].get("model_name")
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=dtype,
        device_map=None
    )
    
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    lora_cfg = LoraConfig(
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(base, lora_cfg)
    model = model.to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        use_fast=True, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def build_tokenize_fn(tokenizer, prompt: str, max_length: int):
    """
    Tokenize per-example, truncating the prompt to leave room for the target.
    Returns variable-length lists; collator will handle padding.
    """
    # Ensure we can pad later
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        statements: List[str] = [str(s).strip() for s in batch["statement"]]
        verdicts:   List[str] = [str(v).strip() for v in batch["verdict"]]
        
        input_ids_all: List[List[int]] = []
        label_all: List[List[int]] = []

        orig_side = tokenizer.truncation_side
        tokenizer.truncation_side = "right"
        
        for s, v in zip(statements, verdicts):
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


def data_processing(cfgs, tokenizer, split_name: str = "training"):
    data_path = cfgs["data"][split_name]
    ds = load_dataset("json", data_files=data_path)
    tokenize_fn = build_tokenize_fn(tokenizer, cfgs["prompt"]["template"], int(cfgs["data"].get("max_length", 0)))
    
    if split_name == "training":
        split = ds["train"].train_test_split(test_size=0.15, seed=cfgs.get("seed", 7))
        dataset = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
            })
        cols_to_drop = dataset["train"].column_names
        return dataset.map(
            tokenize_fn, 
            batched = True,
            num_proc=4,
            remove_columns=cols_to_drop,
            desc="Tokenizing",
        )
    else: 
        return ds.map(
            tokenize_fn, 
            batched = True,
            num_proc=4,
            desc="Tokenizing",
        )


def build_metrics_computer(tokenizer):
    """
    Exact-string accuracy for the target only. Convert from logits -> ids when needed and compare on the label positions (mask != -100).
    """
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if preds.ndim == 3: 
            pred_ids = preds.argmax(-1) 
        else:
            pred_ids = preds 

        labels = np.array(labels)
        pred_ids = np.array(pred_ids)
        
        y_true: List[str] =  []
        y_pred: List[str] = []
        
        for i in range(labels.shape[0]):
            label_row = labels[i]
            pred_row = pred_ids[i]

            shifted_labels = label_row[1:]
            shifted_preds = pred_row[:-1]

            mask = shifted_labels != -100
            if not np.any(mask):
                continue

            true_ids = shifted_labels[mask]
            pred_span_ids = shifted_preds[mask]
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
                metrics[f"per_class/recall/{lab}"]    = float(rec[i])
                metrics[f"per_class/f1/{lab}"]        = float(f1[i])
                metrics[f"per_class/support/{lab}"]   = float(supp[i])
            
        else:
            metrics = {
                "accuracy": accuracy,
                "macro_f1": 0.0,
                "cohen_kappa": 0.0,
                "recognized_ratio": 0.0,
            }
    
    
        return metrics
    
    return compute_metrics


def write_json(metrics: Dict[str, float], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    LOGGER.info("Persisted metrics to %s", path)


def save_predictions(predictions: np.ndarray, labels: np.ndarray, statements: List[str],
    tokenizer, path: Path, ) -> None:
    """Persist decoded predictions alongside the original statement and label."""

    preds = np.asarray(predictions)
    lbls = np.asarray(labels)

    if preds.ndim == 3:
        pred_token_ids = preds.argmax(-1)
    else:
        pred_token_ids = preds

    results: List[Dict[str, str]] = []
    for pred_row, label_row, statement in zip(pred_token_ids, lbls, statements):
        mask = label_row != -100
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        label_ids = label_row[idx]

        pred_idx = idx - 1
        valid = pred_idx >= 0
        label_ids = label_ids[valid]
        pred_idx = pred_idx[valid]
        if len(pred_idx) == 0:
            continue

        pred_ids = pred_row[pred_idx]

        true_text = tokenizer.decode(label_ids, skip_special_tokens=True).strip().lower()
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip().lower()

        results.append(
            {
                "statement": statement,
                "verdict": true_text,
                "prediction": pred_text,
            }
        )


    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")
    LOGGER.info("Saved %s prediction rows to %s", len(results), path)


def train(model, tokenizer, cfgs: Dict[str, Any]):

    mlflow_cfg = MLflowSettings(**cfgs["mlflow"]) 
    dataset = data_processing(cfgs, tokenizer, "training")
    compute_metrics = build_metrics_computer(tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    
    out_dir = Path(cfgs.get("trainer", {}).get("output_dir", "results/ar-qwen-mini")).resolve()
    
    args = TrainingArguments(   
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        bf16=bf16_ok,
        fp16=(fp16_ok and not bf16_ok),
        report_to=["mlflow"] if mlflow_cfg.enable else [],
        run_name=mlflow_cfg.resolved_run_name(),
        remove_unused_columns=True,
        seed=cfgs["seed"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    trainer.add_callback(EarlyStoppingCallback())
    
    numeric_metrics: Dict[str, float] = {}
    
    try:
        if mlflow_cfg.enable:
            mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
            mlflow.set_experiment(mlflow_cfg.exp_name)
            LOGGER.info("MLflow tracking enabled (experiment=%s)", mlflow_cfg.exp_name)
            with mlflow.start_run(run_name=mlflow_cfg.resolved_run_name()):
                LOGGER.info("Starting training")
                mlflow.log_params({
                    "model_name": cfgs["model"].get("model_name", ""),
                    "epochs": args.num_train_epochs,
                    "train_bs_per_device": args.per_device_train_batch_size,
                    "eval_bs_per_device": args.per_device_eval_batch_size,
                    "lr": args.learning_rate,
                    "bf16": args.bf16,
                    "fp16": args.fp16,
                    "max_length": int(cfgs["data"].get("max_length", 0)),
                    "seed": cfgs.get("seed", 7),
                })
                
                train_output = trainer.train()
                train_metrics = train_output.metrics or {}
                
                _mlflow_log_metrics_renamed(train_metrics)
                metrics = trainer.evaluate()                 # includes eval_loss + compute_metrics()
                _mlflow_log_metrics_renamed(metrics)
                numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                write_json(numeric_metrics, out_dir / "training_metrics.json")
                
                # log to directory
                for path, target in [
                    (out_dir / "training_metrics.json", "metrics"),
                ]:
                    try:
                        mlflow.log_artifact(str(path), artifact_path=target)
                        LOGGER.info(f"Logged {path} to {target}")
                    except Exception as e:
                        LOGGER.warning(f"Failed to log {path}: {e}")

        else:
            LOGGER.info("Starting training without MLflow tracking...")
            trainer.train()
            metrics = trainer.evaluate()
            numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            write_json(numeric_metrics, out_dir / "training_metrics.json")
        
    except Exception as e:
        LOGGER.error(f"Training failed: {e}")
        raise
    
    return trainer


def post_eval(trainer, tokenizer, cfgs, split_name: str = "training"):
    # post training testing.
    dataset = data_processing(cfgs, tokenizer, split_name)
    predictions_output = trainer.predict(dataset["train"])

    preds = predictions_output.predictions
    labels = predictions_output.label_ids

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    preds = np.asarray(preds)
    labels = np.asarray(labels) if labels is not None else None

    if preds.ndim == 3:
        preds = preds.argmax(-1)

    results: List[str] = []
    if labels is not None:
        for pred_row, label_row in zip(preds, labels):
            label_row = np.asarray(label_row)
            mask = label_row != -100
            if not np.any(mask):
                results.append("")
                continue

            shifted_labels = mask[1:]
            shifted_preds = pred_row[:-1]
            target_ids = shifted_preds[shifted_labels]
            text = tokenizer.decode(target_ids, skip_special_tokens=True).strip().lower()
            results.append(text)
    else:
        decoded = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        results = [text.strip().lower() for text in decoded]

    eval_split = dataset["train"]
    statements = list(eval_split["statement"]) if "statement" in eval_split.column_names else []
    verdicts = list(eval_split["verdict"]) if "verdict" in eval_split.column_names else []

    print("about to run the document saving part...")
    if statements and verdicts:
        num_rows = min(len(statements), len(verdicts), len(results))
        output_dir = Path("results/ar-qwen-mini").resolve()
        output_path = output_dir / f"{split_name}_predictions.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("opening file to save now...")
        with output_path.open("w", encoding="utf-8") as fh:
            for idx in range(num_rows):
                row = {
                    "statement": statements[idx],
                    "verdict": verdicts[idx],
                    "prediction": results[idx],
                }
                fh.write(json.dumps(row) + "\n")
        LOGGER.info("Saved %s predictions to %s", num_rows, output_path)
    else:
        LOGGER.warning("Skipping prediction export; statements or verdicts missing")

    # return results
    # eval_statements = (
    #     list(dataset["test"]["statement"])
    #     if "statement" in dataset["validation"].column_names
    #     else []
    # )
    # if predictions_output.label_ids is not None and eval_statements:
    #     save_predictions(
    #         predictions_output.predictions,
    #         predictions_output.label_ids,
    #         eval_statements,
    #         tokenizer,
    #         out_dir / "predictions.jsonl",
    #     )
    #     if mlflow_cfg.enable:
    #         try:
    #             mlflow.log_artifact(str(out_dir / "predictions.jsonl"), artifact_path="predictions")
    #         except Exception as e:
    #             LOGGER.warning("Failed to log predictions: %s", e)
    # else:
    #     LOGGER.warning("Skipping prediction export; labels or statements missing")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/base.yaml")
    args = parser.parse_args()
    
    setup_logging()
    LOGGER.info("Logging setup complete")
    CONFIG_DIR = Path(args.config).resolve()
    cfgs = _read_config(CONFIG_DIR)
    model, tokenizer = load_model(cfgs)
    trained = train(model, tokenizer, cfgs)
    post_eval(trained, tokenizer, cfgs, "testing")
    
    
if __name__ == "__main__":
    main()
