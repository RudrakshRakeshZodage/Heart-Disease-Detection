import os
import numpy as np
import librosa
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

AUDIO_PATH = "../dataset/training_data"
LABELS = ['normal', 'murmur', 'unknown']

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = []

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.append(np.mean(mfcc.T, axis=0))

        # Chroma
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.append(np.mean(chroma.T, axis=0))

        # Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=6, fmin=50)
        features.append(np.mean(spec_contrast.T, axis=0))

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr.T, axis=0))

        # RMS
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms.T, axis=0))

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid.T, axis=0))

        return np.hstack(features)

    except Exception as e:
        print(f"⚠️ Error: {file_path} - {e}")
        return None

data = []
labels = []

print("🔍 Extracting features...")
for label in LABELS:
    folder_path = os.path.join(AUDIO_PATH, label)
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".wav"):
            full_path = os.path.join(folder_path, file)
            feats = extract_features(full_path)
            if feats is not None:
                data.append(feats)
                labels.append(label)

X = np.array(data)
y = np.array(labels)

print(f"\n✅ Feature matrix shape: {X.shape}")
print(f"🔢 Classes: {set(y)}")

# Oversampling using SMOTE
print("🔄 Balancing with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=400, max_depth=40, random_state=42)
model.fit(X_train, y_train)

# Save
with open("heart_audio_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
