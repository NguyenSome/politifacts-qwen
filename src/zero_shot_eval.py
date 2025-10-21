import argparse
import torch, numpy as np
import yaml, json

from typing import List
from datetime import date
from datasets import load_dataset
from sklearn.metrics import f1_score, cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", message="The following generation flags are not valid") 

LABELS = ["pants-fire","false","mostly-false","half-true","mostly-true","true"]
label2id = {l:i for i,l in enumerate(LABELS)}
d = date.today()
date_str = d.strftime("%d%m%y")

def predict_label(model, tok, claim: str) -> str:
    prompt = (
        "Classify the following statement with one label only, "
        "chosen from: pants-fire, false, mostly-false, half-true, mostly-true, true.\n"
        f"Statement: {claim}\n"
        "Answer with only the label, nothing else:\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        out = model.generate(
            **inputs, 
            max_new_tokens=5,  # Reduced to prevent extra generation
            eos_token_id=tok.eos_token_id,
            do_sample=False,  # Use greedy decoding for more consistent results
            num_beams=1,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=False, 
            output_scores=False,
        ) 
    
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = out[0][input_length:]
    gen = tok.decode(generated_tokens, skip_special_tokens=True).strip().lower()
    
    return gen

def zero_shot_baseline(model, tok, data, N):
    rows = []
    y_true: List[str] = []
    y_pred: List[str] = []
    
    for i, ex in enumerate(data.select(range(N))):
        try:
            test_lab = ex["verdict"]
            pred_lab = predict_label(model, tok, ex["statement"])
            true_id = label2id.get(test_lab, -1)
            pred_id = label2id.get(pred_lab, -1)
            
            y_true.append(true_id)
            y_pred.append(pred_id)
            
            rows.append({"true": test_lab, "pred": pred_lab, "statement": ex["statement"]})
            
        except Exception as e:
            rows.append({"true": ex.get("verdict", None), "pred": "error", "statement": ex.get("statement", None), "error": str(e)})
            y_true.append(-1)
            y_pred.append(-1)


    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mask = (y_pred != -1) & (y_true != -1)
    coverage = float(mask.mean()) if len(mask) else 0.0
    
    acc_overall = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    acc_covered = float((y_true[mask] == y_pred[mask]).mean()) if mask.any() else 0.0
    
    # MAE
    if mask.any():
        mae = float(np.abs(y_pred[mask] - y_true[mask]).mean())
    else:
        mae = 0.0
        
    # F1 and Quadratic weighted Kappa
    if mask.any():
        num_labels = len(LABELS)
        labels_idx = list(range(num_labels))
        macro_f1 = float(f1_score(y_true[mask], y_pred[mask],
                                  average="macro", labels=labels_idx, zero_division=0))
        qwk = float(cohen_kappa_score(y_true[mask], y_pred[mask], weights="quadratic"))
    else:
        macro_f1, qwk = 0.0, 0.0
        
    metrics = {
        "size": N,
        "coverage": coverage,
        "accuracy_overall": acc_overall,
        "accuracy_on_covered": acc_covered,
        "mae_on_covered": mae,
        "macro_f1_on_covered": macro_f1,
        "qwk_on_covered": qwk,
    }
    
    path = Path(f"results/baseline_predictions_{date_str}.jsonl").resolve()    
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
            
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    
    CONFIG_DIR = Path(args.config).resolve()
    with open(CONFIG_DIR) as f:
        cfg = yaml.safe_load(f)
    
    tok = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["model_name"], dtype="auto", device_map="auto")
    model.eval()
    
    data_path = cfg["data"]["testing"]
    ds = load_dataset("json", data_files= str(data_path))
    ds = ds["train"].select_columns(["verdict", "statement"])
    
    metrics = zero_shot_baseline(model, tok, ds, len(ds))
    
    print("finished")
    print(metrics)

if __name__ == "__main__":
    main()