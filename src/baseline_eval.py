import os, re, torch, numpy as np, pandas as pd
import yaml
from . import paths

from datasets import load_dataset
from sklearn.metrics import f1_score, cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForCausalLM

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", message="The following generation flags are not valid") 


LABELS = ["pants-fire","false","mostly-false","half-true","mostly-true","true"]
label2id = {l:i for i,l in enumerate(LABELS)}


def predict_label(claim: str) -> str:
    messages = [
        {"role": "system", "content":
        "You are a fact-checking classifier. Respond with exactly one label from: "
        "[pants-fire, false, mostly-false, half-true, mostly-true, true]. "
        "Do not add punctuation or explanation."},
        {"role": "user", "content": f'Statement: "{claim}"\nPick the most appropriate verdict from {LABELS}'}
    ]
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # plain-text fallback prompt
        prompt = (
        "System: You are a fact-checking classifier. Respond with exactly one label from: "
        "[pants-fire, false, mostly-false, half-true, mostly-true, true]. No punctuation, no explanation.\n"
        f'User: Statement: "{claim}"\nPick the most appropriate verdict from {LABELS}'
        )
        
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4, eos_token_id=tok.eos_token_id) 
        
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    gen_lower = gen.lower()
    
    for lab in LABELS:
        if gen_lower.startswith(lab): 
            return lab
        
    m = re.search(r"(pants-fire|false|mostly-false|half-true|mostly-true|true)", gen_lower)
    return m.group(1) if m else "unknown"  # counted as wrong


def zero_shot_baseline(data, N):
    
    rows = []
    y_true, y_pred = [], []
    
    for ex in data.select(range(N)):
        try:
            test_lab = ex["verdict"]
            pred_lab = predict_label(ex["statement"])
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

    # Save per-example outputs for inspection
    df = pd.DataFrame(rows)
    out_csv = "predictions.csv"
    df.to_csv(out_csv, index=False)
    
    return metrics

def main():
    
    with open(paths.CONFIG_DIR / "eval.yaml") as f:
        cfg = yaml.safe_load(f)
    
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    ds = load_dataset("json", data_files="test.json")
    ds = ds["train"].select_columns(["verdict", "statement"])
    
    metrics = zero_shot_baseline(ds, len(ds))
    
    print("finished")
    print(metrics)

if __name__ == "__main__":
    main()