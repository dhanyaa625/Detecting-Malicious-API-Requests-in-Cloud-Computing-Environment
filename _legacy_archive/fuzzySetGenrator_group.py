from utility import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

# Load dataset
dataset = pd.read_csv("data/merge_csv_20240704.csv")
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# Generalize label replacement
dataset['label'] = dataset['label'].replace(to_replace=r'benign.*', value='benign', regex=True)

# Remove classes with small numbers
value_counts = dataset['label'].value_counts()
to_remove = value_counts[value_counts <= 50].index
df = dataset[~dataset.label.isin(to_remove)]
value_counts = df['label'].value_counts()
print(value_counts)
print(len(df))

# Data processing
x_train = df

num_columns = 16
for column_i in range(54, 455, num_columns):
    print(x_train.columns[column_i:column_i + num_columns])
    joint_columns_data = x_train.iloc[:, column_i:column_i + num_columns].values.tolist()
    print(joint_columns_data[0])

    k_values = range(1, 80)  # Try 1 to 79 clusters

    # Initialize lists to store results
    wcss_scores = []  # For the Elbow Method
    silhouette_scores = []  # For Silhouette Analysis

    # Calculate WCSS (Within-Cluster Sum of Squares) for the Elbow Method
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(joint_columns_data)  # Use the entire dataset
        wcss_scores.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, wcss_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Elbow Method)')
    plt.title('Elbow Method')

    # Calculate Silhouette Scores for Silhouette Analysis
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(joint_columns_data)  # Use the entire dataset and get labels
        if len(set(labels)) > 1:  # Ensure there is more than one cluster
            silhouette_avg = silhouette_score(joint_columns_data, labels)  # Calculate silhouette score
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(-1)  # Append -1 if only one cluster found

    # Plot the Silhouette Analysis graph
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score (Silhouette Analysis)')
    plt.title('Silhouette Analysis')
    filename = f'plots/{column_i}_{x_train.columns[column_i]}.jpg'
    plt.savefig(filename)
    plt.tight_layout()
    plt.show()

    # Find the optimal number of clusters based on the Elbow Method and Silhouette Analysis
    optimal_k_elbow = wcss_scores.index(min(wcss_scores)) + 2  # Add 2 to account for starting from k=2
    optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 to account for starting from k=2

    print(f'Optimal number of clusters (Elbow Method): {optimal_k_elbow}')
    print(f'Optimal number of clusters (Silhouette Analysis): {optimal_k_silhouette}')

    # Dummy data for student marks (features)
    joint_columns_data = np.array(joint_columns_data)

    # Number of clusters (fuzzy sets)
    n_clusters = 15

    # Apply K-Means clustering to determine initial cluster centers
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(joint_columns_data)
    cluster_centers = kmeans.cluster_centers_

    # Calculate membership parameters a, b, and c for each feature
    a_values = cluster_centers - np.std(joint_columns_data, axis=0)  # Adjust the fraction as needed
    c_values = cluster_centers + np.std(joint_columns_data, axis=0)  # Adjust the fraction as needed
    b_values = cluster_centers

    # Plot the membership functions for each feature
    # features = joint_columns_data.shape[1]
    #
    # for feature in range(features):
    #     plt.figure()
    #     x = np.arange(joint_columns_data[:, feature].min() - 5, joint_columns_data[:, feature].max() + 5, 1)
    #     for i in range(n_clusters):
    #         membership = fuzz.trimf(x, [a_values[i, feature], b_values[i, feature], c_values[i, feature]])
    #         plt.plot(x, membership, label=f'Fuzzy Set {i + 1}')
    #
    #     plt.xlabel(f'Feature {feature + 1}')
    #     plt.ylabel('Membership')
    #     plt.legend()
    #     plt.title(f'Triangular Membership Functions for Feature {feature + 1}')
    #     plt.grid(True)
    # plt.show()

    # Print the calculated membership parameters for each feature
    fuzzy_sets = []
    fuzzy_sets_list = []
    for i in range(n_clusters):
        # print(f'Fuzzy Set {i + 1}: a={a_values[i]}, b={b_values[i]}, c={c_values[i]}')
        fuzzy_set = {'set_i': i + 1, 'a': a_values[i], 'b': b_values[i], 'c': c_values[i]}
        fuzzy_set_list = {'set_i': i + 1, 'a': a_values[i].tolist(), 'b': b_values[i].tolist(), 'c': c_values[i].tolist()}
        fuzzy_sets.append(fuzzy_set)
        fuzzy_sets_list.append(fuzzy_set_list)

    data_points_by_fuzzy_set = {}
    label_frequencies_by_fuzzy_set = {}
    context_fuzzy_sets = []


    def belongs_to_fuzzy_set(features, a_list, b_list, c_list):
        # Check if each feature in the list falls within the corresponding range [a, c]
        for feature, a, b, c in zip(features, a_list, b_list, c_list):
            if not (a <= feature <= c):
                return False
        return True


    # Iterate through each fuzzy set
    for fuzzy_set in fuzzy_sets:
        set_i = fuzzy_set['set_i']
        a_list = fuzzy_set['a']
        b_list = fuzzy_set['b']
        c_list = fuzzy_set['c']
        # Filter data points that belong to this fuzzy set based on the features
        data_points_in_fuzzy_set = x_train[
            x_train.iloc[:, column_i:column_i + num_columns].apply(
                lambda x: belongs_to_fuzzy_set(x, a_list, b_list, c_list),
                axis=1
            )
        ]

        # Calculate label frequencies for the filtered data points
        label_frequencies = data_points_in_fuzzy_set['label'].value_counts()

        # Store the data points and label frequencies in dictionaries
        # data_points_by_fuzzy_set[set_i] = data_points_in_fuzzy_set.iloc[:, column_i:column_i + num_columns].values.tolist()
        label_frequencies_by_fuzzy_set[set_i] = label_frequencies
        context_fuzzy_set = {
            'set_i': set_i,
            'label_frequencies_by_fuzzy_set': label_frequencies_by_fuzzy_set[set_i].to_dict()
        }
        context_fuzzy_sets.append(context_fuzzy_set)

    print("context_fuzzy_sets done!")

    context_data = {
            'column_i': column_i,
            'column_name': x_train.columns[column_i],
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'fuzzy_sets': fuzzy_sets_list,
            'context_fuzzy_sets': context_fuzzy_sets
    }
    json_file_path = f'context/data_{column_i}_{x_train.columns[column_i]}.json'

        # Save the dictionary as JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(context_data, json_file)

    print(column_i, " done!")
