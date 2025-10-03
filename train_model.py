import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# --- Define the feature order ---
# This ensures consistency between training and prediction
FEATURE_ORDER = ['temperature', 'heartrate', 'spo2', 'respRate', 'bp_raw', 'ecg_mv']

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('dataset/synthetic_vitals_dataset.csv')

# Use the defined order for features (X)
X = df[FEATURE_ORDER]
y = df['Anomaly_Label']

# 2. Encode text labels into numbers
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# ... (rest of the script is the same) ...

# 3. Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 4. Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Random Forest model
print("Training Base Model 1: Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred_train = rf.predict_proba(X_train_scaled)
rf_pred_test = rf.predict_proba(X_test_scaled)

# 6. Train ANN model
print("Training Base Model 2: Artificial Neural Network...")
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
ann = Sequential()
ann.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
ann.add(Dense(16, activation='relu'))
ann.add(Dense(num_classes, activation='softmax'))
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_train_scaled, y_train_categorical, epochs=50, batch_size=32, verbose=0)
ann_pred_train = ann.predict(X_train_scaled)
ann_pred_test = ann.predict(X_test_scaled)

# 7. Create stacked features for the meta-model
print("Stacking features from base models...")
stacked_features_train = np.hstack((rf_pred_train, ann_pred_train))
stacked_features_test = np.hstack((rf_pred_test, ann_pred_test))

# 8. Train the Meta-Learner
print("Training Meta-Model...")
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(stacked_features_train, y_train)
y_pred = meta_model.predict(stacked_features_test)
stacked_accuracy = accuracy_score(y_test, y_pred)

# 9. Save all model components
print("Saving all model components...")
os.makedirs("model", exist_ok=True)
joblib.dump(rf, "model/rf_model.pkl")
ann.save("model/ann_model.h5")
joblib.dump(meta_model, "model/meta_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

# 10. Save model accuracy
with open("model/accuracy.txt", "w") as f:
    f.write(str(stacked_accuracy))

print(f"\n✅ Hybrid model training complete. Accuracy on test set: {stacked_accuracy * 100:.2f}%")