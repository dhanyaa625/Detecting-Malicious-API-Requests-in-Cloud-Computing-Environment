import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import json
from utility import *
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_instance(data, scaler_path):
    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Scale features
    instance_scaled = scaler.transform(data)

    return instance_scaled


def get_features_for_class(attention_weights, feature_names, actual_label_index):
    attn_numpy = attention_weights[actual_label_index].numpy().flatten()
    features = [[feature_names[i], float(attn_numpy[i])] for i in range(len(feature_names))]
    return features


def generate_explanations_for_class(model, scaler_path, test_df, feature_names, class_label, num_samples=100):
    class_df = test_df[test_df['label'] == class_label]
    selected_samples = shuffle(class_df, random_state=34)[:num_samples]

    explanations = []
    all_index = []
    for index, sample in selected_samples.iterrows():
        all_index.append(index)
        instance = sample.drop(labels=['label', 'filename']).values.reshape(1, -1)
        instance_scaled = preprocess_instance(instance, scaler_path)
        attention_weights = model.layers[1](instance_scaled)
        predicted_label = np.argmax(model.predict(instance_scaled), axis=-1)[0]
        features = get_features_for_class(attention_weights, feature_names, predicted_label)  # Assuming actual class index is 0
        explanations.append({
            'Sample Index': index,
            'Actual Label': class_label,
            #'Actual Label': str(predicted_label),
            'Features': features
        })
    selected_samples_with_label = {'Actual Label': str(class_label), 'selected_samples': all_index}
    return explanations, selected_samples_with_label


def main(model_path, scaler_path, test_data_path, feature_names_path, label_encoder_path):
    model = load_model(model_path)
    test_df = pd.read_csv(test_data_path)
    with open(feature_names_path, 'r') as f:
        feature_names = f.read().splitlines()

    unique_classes = test_df['label'].unique()
    all_explanations = []

    all_selected_samples = []
    for class_label in unique_classes:
        class_explanations, selected_samples_with_label = generate_explanations_for_class(model, scaler_path, test_df, feature_names, class_label)
        all_explanations.extend(class_explanations)
        all_selected_samples.append(selected_samples_with_label)

    # Save the explanations as JSON
    json_file_path = "prospect/pacx_explanations.json"
    json_file_path_all_selected_samples = "prospect/all_selected_samples.json"
    with open(json_file_path, 'w') as json_file:
        json.dump(all_explanations, json_file, indent=4)
    with open(json_file_path_all_selected_samples, 'w') as json_file2:
        json.dump(all_selected_samples, json_file2, indent=4)


if __name__ == "__main__":
    print(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_path = "final_outputs_20240705_032407"
    MODEL_PATH = f'model/{output_path}/model_'
    TEST_DATA_PATH = f'model/{output_path}/test_data.csv'
    SCALER_PATH = f'model/{output_path}/scaler.pkl'
    FEATURE_NAMES_PATH = f'data/features.txt'
    LABEL_ENCODER_PATH = f'model/{output_path}/label_encoder.pkl'  # Path to your saved LabelEncoder

    main(MODEL_PATH, SCALER_PATH, TEST_DATA_PATH, FEATURE_NAMES_PATH, LABEL_ENCODER_PATH)
    print(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
