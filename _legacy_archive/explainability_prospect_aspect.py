import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import json
import re
from sklearn.preprocessing import LabelEncoder


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_single_instance(data, scaler_path, features_to_drop=['label', 'filename']):
    # Drop non-feature columns
    instance = data.drop(columns=features_to_drop)

    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Scale features
    instance_scaled = scaler.transform(instance)

    return instance_scaled

def get_top_features_per_class(attention_weights_list, feature_names, label_encoder, top_k=10):
    top_features = {}

    for i, attn in enumerate(attention_weights_list):
        attn_numpy = attn.numpy().flatten()
        feature_attention_dict = {}

        for idx, feature_name in enumerate(feature_names):
            #base_name = '_'.join(feature_name.split('_')[:-1])  # Aggregate similar features
            #base_name = re.sub(r'(_\d+)$', '_0', feature_name)
            base_name = feature_name
            if base_name not in feature_attention_dict:
                feature_attention_dict[base_name] = []
            feature_attention_dict[base_name].append(attn_numpy[idx])

        aggregated_attention = {base_name: np.mean(attn_values) for base_name, attn_values in feature_attention_dict.items()}
        sorted_features = sorted(aggregated_attention.items(), key=lambda item: item[1], reverse=True)

        # Use the label encoder to get the actual label name for the class
        class_label = label_encoder.inverse_transform([i])[0]

        # Ensure top_features is correctly updated with class_label as the key
        top_features[str(class_label)] = [(feature, float(weight)) for feature, weight in sorted_features[:top_k]]

    return top_features



def main(model_path, scaler_path, test_data_path, feature_names_path, label_encoder_path, sample_index):
    model = load_model(model_path)

    with open(feature_names_path, 'r') as f:
        feature_names = f.read().splitlines()

    test_df = pd.read_csv(test_data_path)
    sample_instance = test_df.iloc[[sample_index]]
    actual_label = sample_instance['label'].values[0]
    instance_scaled = preprocess_single_instance(sample_instance, scaler_path)

    le = joblib.load(label_encoder_path)
    prediction = model.predict(instance_scaled)
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = le.inverse_transform([predicted_label_index])[0]
    label_probabilities = prediction.flatten().tolist()

    conditional_attentions_list = model.layers[1](instance_scaled)

    top_features_per_class = get_top_features_per_class(conditional_attentions_list, feature_names, le, top_k=10)

    result_dict = {
        'Actual Label': actual_label,
        'Predicted Label': predicted_label,
        'Label Probabilities': {str(le.inverse_transform([i])[0]): float(prob) for i, prob in
                                enumerate(label_probabilities)},
        'Top 10 Features Per Class': top_features_per_class
    }
    print(result_dict)
    return result_dict

# Usage remains the same as before

if __name__ == "__main__":
    output_path = "final_outputs_20240704_143756"
    sample_index = 747
    MODEL_PATH = f'model/{output_path}/model_'
    TEST_DATA_PATH = f'model/{output_path}/test_data.csv'
    SCALER_PATH = f'model/{output_path}/scaler.pkl'
    FEATURE_NAMES_PATH = f'data/features.txt'
    LABEL_ENCODER_PATH = f'model/{output_path}/label_encoder.pkl'  # Path to your saved LabelEncoder

    result_dict = main(MODEL_PATH, SCALER_PATH, TEST_DATA_PATH, FEATURE_NAMES_PATH, LABEL_ENCODER_PATH, sample_index)
    # Save the dictionary as JSON
    json_file_path = f'model/{output_path}/prospect_aspect_{sample_index}.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(result_dict, json_file)

