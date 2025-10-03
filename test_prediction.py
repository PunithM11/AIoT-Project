import numpy as np
import joblib
from keras.models import load_model

# Load trained models
rf = joblib.load('model/rf_model.pkl')
ann = load_model('model/ann_model.h5')
meta = joblib.load('model/meta_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Load model accuracy saved during training
with open("model/accuracy.txt", "r") as f:
    model_accuracy = float(f.read().strip())

# Class labels
class_labels = ['Normal', 'Hypoxia', 'Fever', 'Bradycardia', 'Tachycardia', 'Hypertension', 'Hypotension', 'Arrhythmia']

# Prompt user to enter sensor values
print("Enter the following vital signs:")

try:
    temperature = float(input("🌡️  Temperature (°C): "))
    heartrate = float(input("❤️  Heart Rate (bpm): "))
    spo2 = float(input("🩸 SpO₂ (%): "))
    bp_sys = float(input("💉 Systolic BP (mmHg): "))
    resp_rate = float(input("💨 Respiratory Rate (breaths/min): "))
    ecg_avg = float(input("🧠 ECG Avg RMS (mV): "))

    sample = [temperature, heartrate, spo2, bp_sys, resp_rate, ecg_avg]
    X = np.array(sample).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Get base predictions
    rf_pred = rf.predict_proba(X_scaled)
    ann_pred = ann.predict(X_scaled)

    # Combine into stacked input
    stacked_input = np.hstack((rf_pred, ann_pred))
    final_pred = meta.predict(stacked_input)[0]
    confidence = np.max(meta.predict_proba(stacked_input)) * 100

    print("\n🔎 Prediction Summary:")
    print(f"🩺 Predicted Anomaly : {class_labels[final_pred]}")
    print(f"🔍 Confidence Level   : {confidence:.2f}%")
    print(f"📊 Model Accuracy     : {model_accuracy * 100:.2f}%")

except Exception as e:
    print("⚠️  Invalid input. Please enter numeric values only.")
    print("Error:", e)