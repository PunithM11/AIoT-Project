import csv
import os
import random
import numpy as np

# --- Configuration ---
NUM_ROWS = 10000
OUTPUT_DIR = "dataset"
FILENAME = os.path.join(OUTPUT_DIR, "synthetic_vitals_dataset.csv")

# 'patient_id' has been removed from the header
HEADERS = [
    "temperature", "heartrate", "spo2", "respRate", 
    "bp_raw", "ecg_mv", "Anomaly_Label"
]

LABELS = [
    'Normal', 'Hypoxia', 'Fever', 'Bradycardia', 'Tachycardia', 
    'Hypertension', 'Hypotension', 'Arrhythmia'
]

def generate_vitals():
    """Generates a single row of vital signs with a corresponding anomaly label."""
    
    # Choose a label. Make 'Normal' more common than other anomalies.
    label = random.choices(LABELS, weights=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], k=1)[0]
    
    # --- Generate vitals based on the chosen label ---
    
    # Normal baseline values
    temp = round(random.uniform(36.5, 37.5), 2)
    hr = random.randint(60, 100)
    spo2 = random.randint(95, 100)
    resp = random.randint(12, 20)
    bp = random.randint(90, 120) # Simulating systolic as the main bp_raw value
    ecg = round(random.uniform(-0.5, 1.5), 4)

    # Introduce anomalies
    if label == 'Fever':
        temp = round(random.uniform(38.0, 40.5), 2)
        hr = random.randint(100, 120)
        resp = random.randint(20, 24)
    
    elif label == 'Hypoxia':
        spo2 = random.randint(85, 94)
        hr = random.randint(100, 130)
        resp = random.randint(22, 30)
        
    elif label == 'Bradycardia':
        hr = random.randint(40, 59)

    elif label == 'Tachycardia':
        hr = random.randint(101, 160)

    elif label == 'Hypertension':
        bp = random.randint(140, 190)

    elif label == 'Hypotension':
        bp = random.randint(70, 89)
        hr = random.randint(90, 120)
        
    elif label == 'Arrhythmia':
        hr = hr + random.randint(-20, 20)
        ecg = round(random.uniform(-1.5, 2.5), 4)

    # 'patient_id' has been removed from the returned dictionary
    return {
        "temperature": temp,
        "heartrate": hr,
        "spo2": spo2,
        "respRate": resp,
        "bp_raw": bp,
        "ecg_mv": ecg,
        "Anomaly_Label": label
    }

def create_dataset():
    """Creates the dataset and saves it to a CSV file."""
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating {NUM_ROWS} rows of synthetic data...")

    with open(FILENAME, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        
        # Write the header row
        writer.writeheader()
        
        # Write the data rows
        for i in range(NUM_ROWS):
            writer.writerow(generate_vitals())
            if (i + 1) % 1000 == 0:
                print(f"  ... {i + 1} rows generated.")

    print(f"\n✅ Success! Dataset saved to '{FILENAME}'")

# --- Main execution block ---
if __name__ == "__main__":
    create_dataset()