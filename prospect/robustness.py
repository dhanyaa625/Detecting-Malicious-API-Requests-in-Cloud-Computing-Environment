"""
Robustness check: how distinguishable are two different malware families'
PAC-X feature explanations from each other (via MMD), and does that
separation hold up as more samples are compared? Reads
prospect/pacx_explanations.json next to this script -- see consistency.py
for why that must be resolved relative to the script, not the CWD.

Usage: python prospect/robustness.py
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
        shap_data = json.load(file)

    # Extract features from SHAP explanations, grouped by class
    class_features = {}
    for item in shap_data:
        features = np.array([value for _, value in item["Features"]])
        class_features.setdefault(item["Actual Label"], []).append(features)

    # MMD between every pair of classes, for growing sample counts (capped by
    # how many samples that class actually has, instead of always assuming 100).
    mmd_scores = []
    class_names = list(class_features.keys())
    for class_a in class_names:
        for class_b in class_names:
            if class_a == class_b:
                continue
            max_n = min(100, len(class_features[class_a]), len(class_features[class_b]))
            for num_files in range(1, max_n + 1):
                data_a = np.array(class_features[class_a][:num_files])
                data_b = np.array(class_features[class_b][:num_files])
                mmd_score = compute_mmd(data_a, data_b)
                mmd_scores.append([class_a, class_b, num_files, mmd_score])

    mmd_scores_df = pd.DataFrame(mmd_scores, columns=["Class 1", "Class 2", "Number of Files", "MMD Score"])
    csv_path = os.path.join(SCRIPT_DIR, "robustness_mmd_scores_pacx.csv")
    mmd_scores_df.to_csv(csv_path, index=False)
    print(f"[+] Wrote {len(mmd_scores_df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
