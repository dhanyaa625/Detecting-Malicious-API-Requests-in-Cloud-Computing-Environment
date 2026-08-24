import json
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load the provided JSON file to analyze the SHAP explanations
output_path = "final_outputs_20240705_032407"
json_file_path = f'pacx_explanations.json'
with open(json_file_path, "r") as file:
    shap_explanations = json.load(file)


# Define a function to compute the Maximum Mean Discrepancy (MMD)
def compute_mmd(X, Y, gamma=0.5):
    """Compute the Maximum Mean Discrepancy (MMD) between two sets of samples X and Y."""
    # Compute the squared distances between each pair of points in the two sets
    XX = euclidean_distances(X, X, squared=True)
    YY = euclidean_distances(Y, Y, squared=True)
    XY = euclidean_distances(X, Y, squared=True)

    # Compute the kernel matrices
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)

    # Compute the MMD score
    mmd_score = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd_score


# Group the SHAP explanations by class/label
class_groups = {}
for explanation in shap_explanations:
    label = explanation["Actual Label"]
    if label not in class_groups:
        class_groups[label] = []
    class_groups[label].append(explanation)

# For each class, calculate the MMD score for 1-1, 2-2, ..., 5-5 samples and store the results
mmd_scores = []
for label, explanations in class_groups.items():
    for i in range(1, 101):
        data1 = []
        data2 = []
        for j, explanation in enumerate(explanations[:i * 2]):  # Selecting 2*i samples, split into two groups
            features = np.array([feature[1] for feature in explanation["Features"]])
            if j % 2 == 0:
                data1.append(features)
            else:
                data2.append(features)

        # Convert lists to NumPy arrays
        data1 = np.array(data1)
        data2 = np.array(data2)

        # Compute the MMD score between the two groups of instances for the current class and sample size
        mmd_score = compute_mmd(data1, data2)
        mmd_scores.append((label, i, mmd_score))

# Convert the results to a Pandas DataFrame and save it as a CSV file
import pandas as pd

mmd_df = pd.DataFrame(mmd_scores, columns=["Label", "Num_Samples", "MMD_Score"])
mmd_csv_path = "consistent_MMD_Scores_packx.csv"
mmd_df.to_csv(mmd_csv_path, index=False)


