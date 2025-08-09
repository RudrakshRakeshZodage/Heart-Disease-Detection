import streamlit as st
import numpy as np
import librosa
import pickle
import os
from tempfile import NamedTemporaryFile

# Page config
st.set_page_config(page_title="Heart Audio Analyzer", layout="wide")
st.title("🩺 Heart Disease Detection from Heartbeat Audio")
st.markdown("Upload your heart sound (.wav) and this app will classify it as **Normal**, **Murmur**, or **Unknown** using a trained AI model.")

# Load model and accuracy
@st.cache_resource
def load_model_and_accuracy():
    with open("heart_audio_model.pkl", "rb") as f:
        model = pickle.load(f)
    accuracy = "94.13%"  # Static accuracy from training log
    return model, accuracy

model, accuracy = load_model_and_accuracy()

# Show accuracy
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### 🎯 Model Accuracy")
    st.metric(label="Accuracy", value=accuracy)

# Upload .wav file
uploaded_file = st.file_uploader("📤 Upload Heartbeat Audio (.wav)", type=["wav"])

# Feature extractor (62 features)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))

    centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rms_mean = np.mean(librosa.feature.rms(y=y))

    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    tempogram_mean = np.mean(tempogram.T, axis=0)[:5]

    features = np.hstack([
        mfcc_mean, chroma_mean,
        zcr_mean, centroid_mean,
        bandwidth_mean, rolloff_mean,
        rms_mean, tempogram_mean
    ])
    return features

# Prediction and UI
if uploaded_file:
    st.markdown("### 🎧 Uploaded Audio:")
    st.audio(uploaded_file, format="audio/wav")

    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path).reshape(1, -1)
        prediction = model.predict(features)[0].lower()
        confidence = np.max(model.predict_proba(features)) * 100

        st.markdown("### 🩻 Prediction Result")
        if prediction == "murmur":
            st.error("**Result: MURMUR**")
        elif prediction == "normal":
            st.success("**Result: NORMAL**")
        elif prediction == "unknown":
            st.warning("**Result: UNKNOWN**")
        else:
            st.info(f"Result: {prediction.upper()}")

        st.markdown(f"### 🔍 Confidence Score: `{confidence:.2f}%`")

        # Suggestions
        st.markdown("---")
        st.markdown("### 🧭 Suggested Next Steps")

        if prediction == "murmur":
            st.warning("⚠️ This audio shows signs of a **heart murmur**.")
            st.markdown("""
- 🧘‍♀️ **Do NOT panic**. Murmurs can be harmless.
- 📞 **Consult a doctor** for evaluation.
- 🧪 Recommended tests:
  - ECG
  - Echocardiogram
  - Chest X-ray or blood tests
- 🔐 Save a copy of this result.

> **Disclaimer:** This tool is educational. Seek a licensed physician's advice.
""")

        elif prediction == "normal":
            st.success("✅ Heart sound appears **normal**.")
            st.markdown("> 👍 Keep maintaining a healthy lifestyle and do regular checkups.")

        elif prediction == "unknown":
            st.warning("😕 Could not confidently classify.")
            st.markdown("> ❗ Try re-uploading with less noise.\n> 📞 Consider a clinical consultation.")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

    finally:
        os.remove(tmp_path)
