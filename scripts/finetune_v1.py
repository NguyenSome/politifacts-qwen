import argparse
import gc
import json
import logging
import logging.config
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
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    action="ignore", message="The following generation flags are not valid"
)

class MemUsageCallback(TrainerCallback):
    def _log_mem(self, where: str):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved() / 1024**3
        print(
            f"[{where}] [GPU {torch.cuda.current_device()}] "
            f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB",
            flush=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        self._log_mem("train_step_end")
        gc.collect()

    def on_evaluate(self, args, state, control, **kwargs):
        # Called once per eval loop
        self._log_mem("on_evaluate_start")
        gc.collect()

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

    base_model = "Qwen/Qwen2.5-3B"

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

    base.config.use_cache = False
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
    Returns variable-length lists; collator will handle padding.
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


def data_processing(cfgs, tokenizer, split_name: str = "training"):
    data_path = cfgs["data"][split_name]
    ds = load_dataset("json",
                      data_files=data_path,
                      keep_in_memory=False,)

    max_length = int(cfgs["data"].get("max_length", 1024))
    tokenize_fn = build_tokenize_fn(
        tokenizer, cfgs["prompt"]["template"], max_length,
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

        def tokenize_for_train(batch: dict[str, list[Any]]) -> dict[str, Any]:
            out = tokenize_fn(batch)
            # We don't need these during training/validation; they only cost RAM.
            out.pop("statement", None)
            out.pop("verdict", None)
            return out

        return dataset.map(
            tokenize_for_train,
            batched=True,
            batch_size=32,
            num_proc=1,
            remove_columns=cols_to_drop,
            desc="Tokenizing (train/val)",
        )
    else:
        cols_to_drop = ds["train"].column_names
        # tokenize_fn returns "statement" and "verdict", so we can remove the original columns
        # Access the "train" split before mapping since load_dataset returns DatasetDict with "train" key
        tokenized = ds["train"].map(
            tokenize_fn,
            batched=True,
            batch_size=32,
            num_proc=1,
            remove_columns=cols_to_drop,
            desc=f"Tokenizing ({split_name})",
        )

        return DatasetDict({split_name: tokenized})


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

        y_true: list[str] = []
        y_pred: list[str] = []

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
            true_text = (
                tokenizer.decode(true_ids, skip_special_tokens=True).strip().lower()
            )
            pred_text = (
                tokenizer.decode(pred_span_ids, skip_special_tokens=True)
                .strip()
                .lower()
            )

            y_true.append(true_text)
            y_pred.append(pred_text)

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


def _manual_training_params() -> dict[str, Any]:
    """Single place to tweak core training loop knobs."""

    return {
            "output_dir": Path("results/ar-qwen-mini").resolve(),
            "logging_steps": 50,   # how often to log training loss
        }


def train(model, tokenizer, cfgs: dict[str, Any], training_runtime: dict[str, Any]):

    mlflow_cfg = MLflowSettings(**cfgs["mlflow"])
    dataset = data_processing(cfgs, tokenizer, "training")
    compute_metrics = build_metrics_computer(tokenizer)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    out_dir = training_runtime["output_dir"]
    log_every_steps = training_runtime["logging_steps"]

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=log_every_steps,
        eval_accumulation_steps=1,

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

        load_best_model_at_end=True,
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
        args=args,
        callbacks=[MemUsageCallback()],
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

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
                mlflow.log_params(
                    {
                        "model_name": cfgs["model"].get("model_name", ""),
                        "epochs": args.num_train_epochs,
                        "train_bs_per_device": args.per_device_train_batch_size,
                        "eval_bs_per_device": args.per_device_eval_batch_size,
                        "lr": args.learning_rate,
                        "bf16": args.bf16,
                        "fp16": args.fp16,
                        "seed": cfgs.get("seed", 7),
                    }
                )

                train_output = trainer.train()
                train_metrics = train_output.metrics or {}
                metrics = trainer.evaluate()  # includes eval_loss + compute_metrics()

                numeric_metrics = {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                all_metrics = {**train_metrics, **numeric_metrics}
                write_json(all_metrics, out_dir / "training_metrics.json")

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
            numeric_metrics = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
            write_json(numeric_metrics, out_dir / "training_metrics.json")

    except Exception as e:
        LOGGER.error(f"Training failed: {e}")
        raise

    return trainer


def post_eval(
    trainer,
    tokenizer,
    cfgs,
    split_name: str = "testing",
    output_dir: Path | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 8,
):
    # post training testing.
    dataset = data_processing(cfgs, tokenizer, split_name)[split_name]

    target_dir = (
        Path(output_dir).resolve()
        if output_dir
        else Path("results/ar-qwen-mini").resolve()
    )
    output_path = target_dir / f"{split_name}_predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = trainer.model
    model.to(model.device)
    model.eval()

    # Extract statement and verdict columns before removing them
    # (data collator can't handle string columns)
    statements = list(dataset["statement"])
    verdicts = list(dataset["verdict"])

    # Remove string columns from dataset before creating DataLoader
    dataset_for_loader = dataset.remove_columns(["statement", "verdict"])

    dataloader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=trainer.data_collator,
        )

    print(f"[post_eval] Writing streamed predictions → {output_path}")

    row_index = 0

    with output_path.open("w", encoding="utf-8") as fh:
        with torch.inference_mode():
            for batch in dataloader:
                # Move tensors to GPU where appropriate
                inputs = {
                    k: v.to(model.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                # Generate predicted tokens
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                # Decode predicted text
                decoded = tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )

                # Write each row immediately
                for pred_text in decoded:
                    row = {
                        "statement": statements[row_index],
                        "verdict": verdicts[row_index],
                        "prediction": pred_text.strip().lower(),
                    }
                    fh.write(json.dumps(row) + "\n")
                    row_index += 1

    LOGGER.info("Saved streamed predictions to %s", output_path)


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
    trained = train(model, tokenizer, cfgs, training_runtime)
    post_eval(
        trained,
        tokenizer,
        cfgs,
        "testing",
        output_dir=training_runtime["output_dir"],
    )


if __name__ == "__main__":
    main()
