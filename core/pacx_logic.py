# label
# 3 3904
# 4 2546
# 2 2100
# 5 1795
# 1 1253

import json
import os
import glob
import re

def load_json(filename):
    """Utility function to load a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)

def get_context_explanation(feature, context_dir, list_of_sets_ids):
    feature = re.sub(r'(_\d+)$', '_0', feature)
    context_explanation = {}
    feature_pattern = f'data_*_{feature}.json'  # Pattern to match files for the feature
    # Use glob to find files that match the pattern
    matching_files = glob.glob(os.path.join(context_dir, feature_pattern))
    for file_path in matching_files:
        context_data = load_json(file_path)

        # Extract the desired label frequencies
        fuzzy_set_explanation = [
            {
                'Fuzzy Set ID': set_data["set_i"],
                'Label Frequencies': set_data["label_frequencies_by_fuzzy_set"]
            }
            for set_data in context_data["context_fuzzy_sets"]
            if set_data["set_i"] in list_of_sets_ids
        ]



        # Add the explanation for the current fuzzy set to the overall context explanation
        context_explanation[f'Fuzzy Set for {feature}'] = fuzzy_set_explanation

    return context_explanation

def generate_pac_x_explanation(prospect_file, context_dir, context_file):
    # Load prospect and aspect data
    prospect_aspect_data = load_json(prospect_file)
    context_file_data = load_json(context_file)

    # Extracting prospects
    actual_label = prospect_aspect_data['Actual Label']
    predicted_label = prospect_aspect_data['Predicted Label']
    label_probabilities = prospect_aspect_data['Label Probabilities']

    # Extracting top features for the predicted label
    top_features = prospect_aspect_data['Top 10 Features Per Class'][str(predicted_label)]

    # Context explanation placeholder
    context_explanations = {}
    print(top_features)
    # Iterate over top features to generate context explanations
    for feature, importance in top_features:
        feature = re.sub(r'(_\d+)$', '_0', feature)
        #feature = feature if feature.endswith('_0') else feature + '_0'
        list_of_sets_ids = context_file_data[feature]
        context_explanations[feature] = get_context_explanation(feature, context_dir, list_of_sets_ids)

    # Compile the PAC-X explanation including the new context explanations
    explanation = {
        'Prospect': {
            'Actual Label': actual_label,
            'Predicted Label': predicted_label,
            'Label Probabilities': label_probabilities
        },
        'Aspect': top_features,
        'Context': context_explanations
    }

    return explanation



if __name__ == "__main__":
    # Example usage -- legacy fuzzy PAC-X explainer, superseded by
    # core/pacx_analyzer.py for the live system. Kept for reference only.
    sample_index = 747
    output_path = "final_outputs_20240704_143756"
    prospect_file = f'model/{output_path}/prospect_aspect_{sample_index}.json'
    context_file = f'model/{output_path}/context_{sample_index}.json'  # context file for particular instance
    context_dir = 'context'  # Directory containing context JSON files

    pac_x_explanation = generate_pac_x_explanation(prospect_file, context_dir, context_file)
    json_file_path = f'model/{output_path}/pac-x_{sample_index}.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(pac_x_explanation, json_file)
    print(json.dumps(pac_x_explanation, indent=4))
