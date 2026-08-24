import json
import os
import pandas as pd
import re

# Directory where JSON files are stored
context_folder = 'context/'

# Function to load JSON data from a file
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Function to determine the top 3 closest fuzzy sets for a given feature
def determine_top_3_sets(feature_json, values):
    set_distances = []

    for fuzzy_set in feature_json['fuzzy_sets']:
        distances = []
        for i, value in enumerate(values):
            a = fuzzy_set['a'][i] if isinstance(fuzzy_set['a'], list) else fuzzy_set['a']
            c = fuzzy_set['c'][i] if isinstance(fuzzy_set['c'], list) else fuzzy_set['c']
            # Calculate distance to the closest bound
            distance = min(abs(value - a), abs(value - c))
            distances.append(distance)

        # Calculate average distance for this set
        average_distance = sum(distances) / len(distances)
        set_distances.append((fuzzy_set['set_i'], average_distance))

    # Sort sets by average distance and select top 3
    top_3_sets = sorted(set_distances, key=lambda x: x[1])[:3]

    return [set_id for set_id, _ in top_3_sets]

# Load all JSON files and create a mapping from feature names to JSON content
feature_json_dict = {}
for filename in os.listdir(context_folder):
    if filename.endswith('.json'):
        feature_name = '_'.join(filename.split('.')[0].split('_')[2:])
        feature_json_dict[feature_name] = load_json_data(os.path.join(context_folder, filename))

# Read CSV using pandas
sample_index = 747
output_path = "final_outputs_20240704_143756"
df = pd.read_csv(f'model/{output_path}/test_data.csv')

# Process only the 6th row
row = df.iloc[sample_index]  # 6th row, considering 0-based indexing
sample_results = {'filename': row['filename']}
for feature in feature_json_dict:
    if feature in row:
        values = [float(row[feature])]  # Convert feature value to float and make it a list for consistency
        top_3_sets = determine_top_3_sets(feature_json_dict[feature], values)
        sample_results[feature] = top_3_sets
    elif any(feature.startswith(f'{feature}_') for feature in row.index):
        # Handle compound features by creating an array of values
        compound_feature_values = [float(row[f'{feature}_{i}']) for i in range(16) if f'{feature}_{i}' in row.index]
        top_3_sets = determine_top_3_sets(feature_json_dict[feature], compound_feature_values)
        sample_results[feature] = top_3_sets

json_file_path = f'model/{output_path}/context_{sample_index}.json'
with open(json_file_path, 'w') as json_file:
    json.dump(sample_results, json_file)
print(sample_results)  # Print results for the 6th row
