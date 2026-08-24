import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import json
import joblib
from scipy.stats import chi2_contingency

# Function to load the saved model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# Function to preprocess the test data
def preprocess_data(test_data_path, scaler_path):
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(['label', 'filename'], axis=1)

    # Correctly load the saved StandardScaler using joblib
    scaler = joblib.load(scaler_path)

    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, test_df['label']


# Function to encode labels
def encode_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    ohe = OneHotEncoder(sparse=False)
    labels_one_hot = ohe.fit_transform(labels_encoded.reshape(-1, 1))
    return labels_encoded, labels_one_hot, le.classes_


# Function to evaluate the model
from scipy.stats import chi2_contingency

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, classes, output_path):
    y_pred = model.predict(X_test)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_test_label = np.argmax(y_test, axis=1)

    results = {}

    # Class-wise metrics
    class_metrics = {}
    for i, class_name in enumerate(classes):
        precision = precision_score(y_test_label, y_pred_label, labels=[i], average='weighted')
        recall = recall_score(y_test_label, y_pred_label, labels=[i], average='weighted')
        f1 = f1_score(y_test_label, y_pred_label, labels=[i], average='weighted')

        class_metrics[str(class_name)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }

    results['class_metrics'] = class_metrics

    # Overall metrics
    for avg in ['micro', 'macro', 'weighted']:
        results[f'{avg}_precision'] = precision_score(y_test_label, y_pred_label, average=avg)
        results[f'{avg}_recall'] = recall_score(y_test_label, y_pred_label, average=avg)
        results[f'{avg}_f1'] = f1_score(y_test_label, y_pred_label, average=avg)

    # ROC AUC
    results['roc_auc_macro'] = roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')

    # Save results
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'model/{output_path}/results_{current_time}.json', 'w') as fp:
        json.dump(results, fp, indent=4)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_label, y_pred_label)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'model/{output_path}/confusion_matrix_{current_time}.png')

    # Detection Time Calculation
    detection_log = {'num_samples': [], 'total_time_sec': []}
    batch_size = 100
    cumulative_time = 0

    for i in range(batch_size, len(X_test) + 1, batch_size):
        start_time = time.time()
        _ = model.predict(X_test[:i])  # Predict up to current batch
        end_time = time.time()

        cumulative_time = end_time - start_time
        detection_log['num_samples'].append(i)
        detection_log['total_time_sec'].append(cumulative_time)

    # Save detection time data to CSV
    detection_df = pd.DataFrame(detection_log)
    detection_df.to_csv(f'model/{output_path}/detection_time_{current_time}.csv', index=False)

    print(f'Detection time CSV saved at: model/{output_path}/detection_time_{current_time}.csv')



# Main function to run the evaluation
def main(model_path, test_data_path, scaler_path, output_path):
    model = load_model(model_path)
    X_test, y_test_labels = preprocess_data(test_data_path, scaler_path)
    y_test_encoded, y_test_one_hot, classes = encode_labels(y_test_labels)
    evaluate_model(model, X_test, y_test_one_hot, classes, output_path)


if __name__ == "__main__":
    # Set paths to your model, test dataset, and scaler
    output_path = "final_outputs_20240705_032407"
    MODEL_PATH = f'model/{output_path}/model_'
    #TEST_DATA_PATH = f'model/{output_path}/test_data.csv'
    #SCALER_PATH = f'model/{output_path}/scaler.pkl'
    TEST_DATA_PATH = f'model/final_without_CANNET_outputs_20240705_033113/test_data.csv'
    SCALER_PATH = f'model/final_without_CANNET_outputs_20240705_033113/scaler.pkl'

    main(MODEL_PATH, TEST_DATA_PATH, SCALER_PATH, output_path)
