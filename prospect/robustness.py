import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import json


# Compute the Maximum Mean Discrepancy (MMD) between two sets of samples X and Y.
def compute_mmd(X, Y, gamma=0.5):
    XX = euclidean_distances(X, X, squared=True)
    YY = euclidean_distances(Y, Y, squared=True)
    XY = euclidean_distances(X, Y, squared=True)

    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)

    mmd_score = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd_score


# Load the provided JSON file to analyze the SHAP explanations
output_path = "final_outputs_20240705_032407"
json_file_path = f'pacx_explanations.json'
with open(json_file_path, "r") as file:
    shap_data = json.load(file)

# Extract features and values from SHAP explanations, and group by class
class_features = {}
for item in shap_data:
    label = item['Actual Label']
    if label not in class_features:
        class_features[label] = []
    features = np.array([value for _, value in item['Features']])
    class_features[label].append(features)

# Calculate MMD scores for different numbers of files and between every pair of classes
mmd_scores = []
for num_files in range(1, 101):
    for class_a in class_features.keys():
        for class_b in class_features.keys():
            if class_a != class_b:
                data_a = np.array(class_features[class_a][:num_files])
                data_b = np.array(class_features[class_b][:num_files])
                mmd_score = compute_mmd(data_a, data_b)
                mmd_scores.append([class_a, class_b, num_files, mmd_score])

# Convert the results into a DataFrame and save as CSV
mmd_scores_df = pd.DataFrame(mmd_scores, columns=['Class 1', 'Class 2', 'Number of Files', 'MMD Score'])
mmd_scores_df.to_csv('robustness_mmd_scores_pacx.csv', index=False)
mmd_scores_df.head()

# Please note: The actual execution of this code snippet might not work here due to the absence of the actual JSON file and environment setup.
# This is a conceptual demonstration and needs to be adapted to your specific environment and data.
