from utility import *
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Name the directory
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = f"model/outputs_binary_{current_time}"

# Create the directory
if not os.path.exists(output):
    os.makedirs(output)

# Load and preprocess the dataset
dataset = pd.read_csv("data/merge_csv_20231005.csv")
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset.drop_duplicates(inplace=True)
dataset['label'] = dataset['label'].str.replace(r'^benign_.+', 'benign', regex=True)
dataset.loc[dataset['label'] != 'benign', 'label'] = 'malware'

# Filter out classes with few samples
value_counts = dataset['label'].value_counts()
to_remove = value_counts[value_counts <= 100].index
dataset = dataset[~dataset['label'].isin(to_remove)]

# Split the data into training and test sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

# Save the test set for later evaluation
test_df.to_csv(f'{output}/test_data.csv', index=False)

# Preprocess the training data
X_train = train_df.drop(['label', 'filename'], axis=1)
y_train = train_df['label']

# Label encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)
label_filename = f'{output}/label_encoder.pkl'
joblib.dump(le, label_filename)

# One-hot encode the target variable
ohe = OneHotEncoder(sparse=False)
y_one_hot = ohe.fit_transform(y_encoded.reshape(-1, 1))

# Standard scale the features
sc = StandardScaler()
X_scaled = sc.fit_transform(X_train)
# Save the scaler object to a .pkl file
scaler_filename = f'{output}/scaler.pkl'
joblib.dump(sc, scaler_filename)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)

# Define the conditional attention layer
class ConditionalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_features):
        super(ConditionalAttentionLayer, self).__init__()
        self.num_features = num_features

    def build(self, input_shape):
        self.dense_layers = [Dense(self.num_features, activation='softmax') for _ in range(len(np.unique(y_encoded)))]
        super(ConditionalAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        attentions = [dense_layer(inputs) for dense_layer in self.dense_layers]
        return attentions

# Define the model with conditional attention
def create_model(input_shape, num_classes):
    inputs = Input(shape=(input_shape,))
    conditional_attention = ConditionalAttentionLayer(num_features=input_shape)(inputs)
    attention_mul = Lambda(lambda x: tf.reduce_sum(x, axis=0))(conditional_attention)
    dense1 = Dense(64, activation='relu')(attention_mul)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(32, activation='relu')(dropout1)
    outputs = Dense(num_classes, activation='softmax')(dense2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model(X_train.shape[1], y_train.shape[1])
history = model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val))

# Save the trained model
model_name = f"{output}/model_"
model.save(f"{model_name}")
print("Model trained and saved as:", model_name)

# Save training history plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.savefig(f'{output}/training_history_.png')

# Load the test set
test_df = pd.read_csv(f'{output}/test_data.csv')
X_test = test_df.drop(['label', 'filename'], axis=1)
y_test = test_df['label']
X_test_scaled = sc.transform(X_test)
y_test_encoded = le.transform(y_test)
y_test_one_hot = ohe.transform(y_test_encoded.reshape(-1, 1))

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
y_pred_label = np.argmax(y_pred, axis=1)
y_test_label = np.argmax(y_test_one_hot, axis=1)

# Calculate precision, recall, and ROC AUC
precision = precision_score(y_test_label, y_pred_label, average='macro')
recall = recall_score(y_test_label, y_pred_label, average='macro')
roc_auc = roc_auc_score(y_test_one_hot, y_pred, multi_class='ovr')

print(f'Precision: {precision}, Recall: {recall}, ROC AUC: {roc_auc}')

# Plot ROC AUC curve
fpr = {}
tpr = {}
thresh ={}
for i in range(len(np.unique(y_encoded))):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_one_hot[:,i], y_pred[:,i])
    plt.plot(fpr[i], tpr[i], linestyle='--', label=f'Class {i} vs Rest')

plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig(f'{output}/roc_curve.png')

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test_label, y_pred_label)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(f'{output}/confusion_matrix.png')
