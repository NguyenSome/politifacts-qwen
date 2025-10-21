import json, random
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n", default = int(50))
args = parser.parse_args()

train_path = Path("data/train.json").resolve()
out_path = Path("data/small_train.json").resolve()

with train_path.open("r", encoding="utf-8") as fh:
    records = [json.loads(line) for line in fh if line.strip()]

if len(records) < int(args.n):
    raise ValueError(f"Not enough records ({len(records)}) to sample 100 entries")

random.seed()
sample = random.sample(records, int(args.n))

with out_path.open("w", encoding="utf-8") as fh:
    for row in sample:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")