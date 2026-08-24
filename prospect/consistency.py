"""
Consistency check: for each malware family, does PAC-X's per-sample feature
explanation stay stable as you compare more and more same-class samples
against each other (via Maximum Mean Discrepancy)? Reads prospect/pacx_explanations.json
(SHAP-style per-sample feature explanations), which lives next to this
script -- previously read via a bare relative filename, which only resolved
correctly if you first `cd`'d into prospect/ yourself; running it the normal
way (`python prospect/consistency.py` from the project root, consistent with
every other script in this project) raised FileNotFoundError.

Usage: python prospect/consistency.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_mmd(X, Y, gamma=0.5):
    """Compute the Maximum Mean Discrepancy (MMD) between two sets of samples X and Y."""
    XX = euclidean_distances(X, X, squared=True)
    YY = euclidean_distances(Y, Y, squared=True)
    XY = euclidean_distances(X, Y, squared=True)

    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)

    return np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)


def main():
    json_file_path = os.path.join(SCRIPT_DIR, "pacx_explanations.json")
    with open(json_file_path, "r") as file:
        shap_explanations = json.load(file)

    # Group the SHAP explanations by class/label
    class_groups = {}
    for explanation in shap_explanations:
        class_groups.setdefault(explanation["Actual Label"], []).append(explanation)

    # For each class, calculate the MMD score for growing same-class sample sizes
    mmd_scores = []
    for label, explanations in class_groups.items():
        max_i = min(100, len(explanations) // 2)
        for i in range(1, max_i + 1):
            data1, data2 = [], []
            for j, explanation in enumerate(explanations[:i * 2]):
                features = np.array([feature[1] for feature in explanation["Features"]])
                (data1 if j % 2 == 0 else data2).append(features)

            mmd_score = compute_mmd(np.array(data1), np.array(data2))
            mmd_scores.append((label, i, mmd_score))

    mmd_df = pd.DataFrame(mmd_scores, columns=["Label", "Num_Samples", "MMD_Score"])
    mmd_csv_path = os.path.join(SCRIPT_DIR, "consistent_MMD_Scores_packx.csv")
    mmd_df.to_csv(mmd_csv_path, index=False)
    print(f"[+] Wrote {len(mmd_df)} rows to {mmd_csv_path}")


if __name__ == "__main__":
    main()


