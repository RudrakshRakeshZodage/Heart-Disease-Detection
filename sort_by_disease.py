import os
import pandas as pd
import shutil

# File paths
CSV_PATH = r"C:\Users\Rudraksh\Desktop\projects\heart_audio_predictor\dataset\training_data.csv"
AUDIO_DIR = r"C:\Users\Rudraksh\Desktop\projects\heart_audio_predictor\dataset\training_data"

# Load the CSV
df = pd.read_csv(CSV_PATH)

# Define murmur label to folder name mapping
label_map = {
    "Present": "murmur",
    "Absent": "normal",
    "Unknown": "unknown",
    "Valve disorder": "valve_disorder",
    "Extrasystole": "extrasystole"
}

# Heart sound positions
positions = ["AV", "MV", "PV", "TV"]

# Create target folders if not exist
for folder in label_map.values():
    os.makedirs(os.path.join(AUDIO_DIR, folder), exist_ok=True)

# Track how many files were moved
moved_count = 0
missing_files = []

# Process each row
for _, row in df.iterrows():
    patient_id = row['Patient ID']
    murmur_status = str(row['Murmur']).strip()

    label_folder = label_map.get(murmur_status)
    if not label_folder:
        print(f"⚠️ Skipping unknown label: '{murmur_status}' for Patient ID {patient_id}")
        continue

    for pos in positions:
        filename = f"{patient_id}_{pos}.wav"
        src_path = os.path.join(AUDIO_DIR, filename)
        dest_path = os.path.join(AUDIO_DIR, label_folder, filename)

        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            moved_count += 1
        else:
            missing_files.append(filename)

# Final summary
print(f"\n✅ Done. Moved {moved_count} audio files into folders.")
if missing_files:
    print(f"⚠️ Missing {len(missing_files)} files:")
    for f in missing_files:
        print(f" - {f}")
