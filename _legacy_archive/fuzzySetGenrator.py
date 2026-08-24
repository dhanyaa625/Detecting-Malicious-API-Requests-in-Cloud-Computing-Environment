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

for column_index in range(53, 54):
    for fuz_i in range(column_index, column_index + 1):
        column_i = fuz_i  # 5 6
        print(x_train.columns[column_i])
        unique_values = x_train[x_train.columns[column_i]].unique()
        print("Unique values in column: ", len(unique_values))

        # Dummy age data
        column_data = np.array(x_train[x_train.columns[column_i]]).reshape(-1, 1)

        # Range of clusters to consider
        k_values = range(2, 80)  # Try 2 to 79 clusters

        # Initialize lists to store results
        wcss_scores = []  # For the Elbow Method
        silhouette_scores = []  # For Silhouette Analysis

        # Calculate WCSS (Within-Cluster Sum of Squares) for the Elbow Method
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(column_data)
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
            kmeans.fit(column_data)
            labels = kmeans.labels_
            if len(set(labels)) > 1:  # Ensure there is more than one cluster
                silhouette_avg = silhouette_score(column_data, labels)
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

        # Dummy data for age
        column_data = np.array(x_train[x_train.columns[column_i]]).reshape(-1, 1)

        # Number of clusters (fuzzy sets)
        n_clusters = min(optimal_k_elbow, optimal_k_silhouette)

        # Apply K-Means clustering to determine initial cluster centers
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(column_data)
        cluster_centers = kmeans.cluster_centers_.squeeze()

        # Calculate membership parameters a, b, and c
        a_values = cluster_centers - np.std(column_data)  # Adjust the fraction as needed
        c_values = cluster_centers + np.std(column_data)  # Adjust the fraction as needed
        b_values = cluster_centers

        fuzzy_sets = []
        # Print the calculated membership parameters
        for i in range(n_clusters):
            fuzzy_set = {'set_i': i + 1, 'a': a_values[i], 'b': b_values[i], 'c': c_values[i]}
            fuzzy_sets.append(fuzzy_set)
            print(i, " fuzzy set done!")

        fuzzy_set_df = pd.DataFrame(fuzzy_sets)
        # Initialize empty dictionaries to store data points and label frequencies for each fuzzy set
        data_points_by_fuzzy_set = {}
        label_frequencies_by_fuzzy_set = {}
        context_fuzzy_sets = []
        # Iterate through each fuzzy set
        for fuzzy_set in fuzzy_sets:
            set_i = fuzzy_set['set_i']
            a = fuzzy_set['a']
            b = fuzzy_set['b']
            c = fuzzy_set['c']

            # Filter data points that belong to this fuzzy set based on membership grades
            data_points_in_fuzzy_set = x_train[
                (x_train[x_train.columns[column_i]] >= a) &
                (x_train[x_train.columns[column_i]] <= c)
            ]

            # Calculate label frequencies for the filtered data points
            label_frequencies = data_points_in_fuzzy_set['label'].value_counts()

            # Store the data points and label frequencies in dictionaries
            label_frequencies_by_fuzzy_set[set_i] = label_frequencies
            context_fuzzy_set = {'set_i': set_i,
                                 'label_frequencies_by_fuzzy_set': label_frequencies_by_fuzzy_set[set_i].to_dict()}
            context_fuzzy_sets.append(context_fuzzy_set)
            print(fuzzy_set, " context fuzzy set done!")

        print("context_fuzzy_sets done!")

        context_data = {
            'column_i': column_i,
            'column_name': x_train.columns[column_i],
            'unique_values': unique_values.tolist(),
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'fuzzy_sets': fuzzy_sets,
            'context_fuzzy_sets': context_fuzzy_sets
        }
        json_file_path = f'context/data_{column_i}_{x_train.columns[column_i]}.json'

        # Save the dictionary as JSON
        with open(json_file_path, 'w') as json_file:
            json.dump(context_data, json_file)
        print(column_i, " done!")
