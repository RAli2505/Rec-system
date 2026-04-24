"""
Reconstruct ablation.json from a partially-completed master ablation log
+ a parallel single-config (-IRT) result file.

Used when we kill the master ablation script after -Confidence finishes
to avoid the redundant -IRT run (parallel one covers it). Master never
writes its ablation.json in that case, so we parse the log instead.

Output: results/xes3g5m/ablation_reconstructed_s42_<ts>/ablation.json
"""

import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

LOG_PATH = "logs/retrain_NUMTAGS858_20260423_065312/ablation_s42.log"
SEED = 42

# 1. Load Full MARS metrics from main pipeline run
mp = sorted(glob.glob(f"results/xes3g5m/xes3g5m_full_s{SEED}_*/metrics.json"),
             key=os.path.getmtime, reverse=True)
mp = [p for p in mp if "20260423" in p]
if not mp:
    sys.exit("No new (today) main pipeline run for seed=42")
with open(mp[0]) as f:
    full_pipe = json.load(f)
full_eval = full_pipe.get("eval_metrics", {})
results = {"Full MARS": dict(full_eval, _source=Path(mp[0]).parent.name)}
print(f"Full MARS from {Path(mp[0]).parent.name}")

# 2. Parse master ablation log for - Prediction, - KG, - Confidence
log = open(LOG_PATH, encoding="utf-8", errors="replace").read()
# Log lines look like:
#   2026-04-23 13:26:33,129 INFO     xes3g5m_ablation:   - Prediction: AUC=0.5077 NDCG=0.5485 P@10=0.5650 MRR=0.6663 Cov=0.0649
# So we match the config name anywhere in the line, not at column 0.
pat = (r"(- (?:Prediction|Knowledge Graph|Confidence|IRT \(Diagnostic\))):\s+"
       r"AUC=([\d.]+)\s+NDCG=([\d.]+)\s+P@10=([\d.]+)\s+MRR=([\d.]+)\s+Cov=([\d.]+)")
for m in re.finditer(pat, log):
    name = m.group(1)
    results[name] = {
        "lstm_auc":     float(m.group(2)),
        "ndcg@10":      float(m.group(3)),
        "precision@10": float(m.group(4)),
        "mrr":          float(m.group(5)),
        "tag_coverage": float(m.group(6)),
        "_source": "reconstructed_from_log",
    }
    print(f"  {name}  AUC={float(m.group(2)):.4f}")

# 3. Load - IRT (Diagnostic) from parallel ablation_one run
one_paths = sorted(
    glob.glob(f"results/xes3g5m/ablation_one_s{SEED}_*/ablation_one.json"),
    key=os.path.getmtime, reverse=True,
)
for p in one_paths:
    with open(p) as f:
        one = json.load(f)
    for k, v in one.items():
        if k not in results:
            v["_source"] = Path(p).parent.name
            results[k] = v
            print(f"  {k}  (from parallel)")

# 4. Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(f"results/xes3g5m/ablation_reconstructed_s{SEED}_{ts}")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "ablation.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2, default=str)
# Bump mtime so newest() picks this over old broken ablation
os.utime(out_file, None)
print(f"\nSaved {out_file}")
print(f"Configs: {list(results.keys())}")
