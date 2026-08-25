"""
Data-mines evidence-weighted API signals for PAC-X from
kaggle_dataset/dynamic_api_call_sequence_per_malware_100_0_306.csv (real
Cuckoo Sandbox API traces), replacing both the small hand-picked suspicious-
API list AND the flat "+15 points per match" scoring in core/pacx_analyzer.py
-- flat scoring treats a near-smoking-gun API (rare in benign, common in
malware) the same as a merely-somewhat-more-common one, which either misses
most malware (small list) or false-positives on ordinary software (larger
list scored the same way). A likelihood-ratio weight fixes this: each API's
contribution is proportional to how much evidence it actually provides.

Proper 3-way split, no peeking:
  - 60% TRAIN: compute per-API P(appears|malware), P(appears|benign), and
    the log-likelihood-ratio weight log2((p_malware+eps)/(p_benign+eps)).
  - 20% VAL: used to pick the single remaining free parameter (the decision
    threshold on the summed weighted score).
  - 20% TEST: touched exactly once, after threshold selection, for the
    number that actually gets reported.

Usage: python scripts/mine_pacx_api_signals.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_kaggle_api_sequences import _API_LEGEND  # noqa: E402

EPS = 0.01


def api_presence_rates(sub_df, t_cols):
    malware_mask = (sub_df["malware"] == 1).values
    benign_mask = (sub_df["malware"] == 0).values
    n_malware = malware_mask.sum()
    n_benign = benign_mask.sum()
    codes_matrix = sub_df[t_cols].values

    rates = {}
    for code, name in enumerate(_API_LEGEND):
        has_api = (codes_matrix == code).any(axis=1)
        p_malware = has_api[malware_mask].mean() if n_malware else 0.0
        p_benign = has_api[benign_mask].mean() if n_benign else 0.0
        rates[name] = {"p_malware": float(p_malware), "p_benign": float(p_benign)}
    return rates


def score_sequence(api_names_present, weights):
    return sum(weights.get(name, 0.0) for name in api_names_present)


def main():
    df = pd.read_csv("kaggle_dataset/dynamic_api_call_sequence_per_malware_100_0_306.csv", low_memory=False)
    t_cols = [f"t_{i}" for i in range(100)]

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_df = df.iloc[idx[:train_end]]
    val_df = df.iloc[idx[train_end:val_end]]
    test_df = df.iloc[idx[val_end:]]
    print(f"[+] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("[+] Computing log-likelihood-ratio weights on TRAIN split...")
    train_rates = api_presence_rates(train_df, t_cols)
    weights = {}
    for name, r in train_rates.items():
        llr = math.log2((r["p_malware"] + EPS) / (r["p_benign"] + EPS))
        weights[name] = round(llr, 4)

    top10 = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("[+] Top 10 highest-evidence APIs (train):")
    for name, w in top10:
        r = train_rates[name]
        print(f"   {name:30s} weight={w:6.2f}  p_malware={r['p_malware']:.3f}  p_benign={r['p_benign']:.3f}")

    def sequence_scores(sub_df):
        codes_matrix = sub_df[t_cols].values
        labels = sub_df["malware"].values
        scores = []
        for row_codes in codes_matrix:
            present = set()
            for c in row_codes:
                try:
                    present.add(_API_LEGEND[int(c)])
                except (ValueError, IndexError):
                    continue
            scores.append(score_sequence(present, weights))
        return np.array(scores), labels

    print("[+] Selecting decision threshold on VAL split (balanced accuracy, "
          "not raw accuracy -- this dataset is 97.5% malware / 2.5% benign, "
          "so raw accuracy trivially rewards 'always predict malicious')...")
    val_scores, val_labels = sequence_scores(val_df)
    best_threshold, best_balanced_acc = None, -1
    for t in np.arange(val_scores.min(), val_scores.max(), 0.5):
        preds = (val_scores > t).astype(int)
        tpr = (preds[val_labels == 1] == 1).mean() if (val_labels == 1).any() else 0.0  # sensitivity
        tnr = (preds[val_labels == 0] == 0).mean() if (val_labels == 0).any() else 0.0  # specificity
        balanced_acc = (tpr + tnr) / 2.0
        if balanced_acc > best_balanced_acc:
            best_balanced_acc, best_threshold = balanced_acc, t
    print(f"[+] Best threshold on VAL: {best_threshold:.2f} (val balanced accuracy {best_balanced_acc*100:.2f}%)")

    print("[+] Final check on TEST split (touched once)...")
    test_scores, test_labels = sequence_scores(test_df)
    test_preds = (test_scores > best_threshold).astype(int)
    test_acc = (test_preds == test_labels).mean()
    fp = int(((test_preds == 1) & (test_labels == 0)).sum())
    fn = int(((test_preds == 0) & (test_labels == 1)).sum())
    n_benign = int((test_labels == 0).sum())
    n_malicious = int((test_labels == 1).sum())
    fp_rate = round(100 * fp / max(1, n_benign), 2)
    fn_rate = round(100 * fn / max(1, n_malicious), 2)
    test_summary = {
        "raw_accuracy": round(float(test_acc) * 100, 2),
        "balanced_accuracy": round(100 - (fp_rate + fn_rate) / 2, 2),
        "false_positive_rate": fp_rate,
        "false_negative_rate": fn_rate,
        "n_tested": len(test_df),
        "n_benign": n_benign,
        "n_malicious": n_malicious,
    }
    print(json.dumps(test_summary, indent=2))

    report = {
        "note": (
            "Log-likelihood-ratio API evidence weights, mined from a 60% TRAIN split of "
            "kaggle_dataset/dynamic_api_call_sequence_per_malware_100_0_306.csv (real Cuckoo Sandbox "
            "traces), threshold chosen on a 20% VAL split, final accuracy measured once on the "
            "untouched 20% TEST split. weight(api) = log2((p_malware+0.01)/(p_benign+0.01))."
        ),
        "decision_threshold": round(float(best_threshold), 2),
        "weights": weights,
        "test_summary": test_summary,
    }
    os.makedirs("data/pacx_mining", exist_ok=True)
    with open("data/pacx_mining/mined_api_signals.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[+] Saved data/pacx_mining/mined_api_signals.json")


if __name__ == "__main__":
    main()
