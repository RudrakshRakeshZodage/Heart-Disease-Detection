# Heart-Disease-Detection

# Overview
- This project is an AI-powered system designed to detect heart disease conditions such as normal, murmur, and unknown from heart sound audio recordings.
- It uses audio signal processing techniques with Librosa to extract multiple features from .wav files, combined with a Random Forest Classifier for classification.
- To handle imbalanced datasets, the project uses SMOTE (Synthetic Minority Oversampling Technique) for balanced model training.

# Features
- Multi-class Classification — Detects normal, murmur, and unknown classes from heart audio.
- Advanced Audio Features — Extracts MFCC, Chroma, Spectral Contrast, Zero Crossing Rate, RMS, and Spectral Centroid.
- Imbalanced Data Handling — Uses SMOTE for dataset balancing.
- Random Forest Classifier — High-performance, interpretable classification model.
- Model Saving — Stores the trained model as a .pkl file for future predictions.

# System Workflow
- Step 1: Load .wav audio files from dataset folders based on their class labels.
- Step 2: Extract MFCC, Chroma, Spectral Contrast, ZCR, RMS, and Spectral Centroid features using Librosa.
- Step 3: Store features and labels in NumPy arrays.
- Step 4: Apply SMOTE to balance the dataset.
- Step 5: Split the dataset into training and testing sets.
- Step 6: Train a Random Forest Classifier with optimized hyperparameters.
- Step 7: Save the trained model using pickle.
- Step 8: Evaluate model performance using Accuracy and Classification Report.

# Technology Stack
Python — Core programming language.
Librosa — Audio processing and feature extraction.
NumPy — Numerical computations and array handling.
Scikit-learn — Random Forest Classifier, train-test splitting, and evaluation.
Imbalanced-learn (SMOTE) — Oversampling for dataset balancing.
Pickle — Model serialization for later use.
TQDM — Progress bar for dataset processing.

# Performance
- Achieves high classification accuracy depending on dataset quality.
- Handles imbalanced datasets effectively using SMOTE.
- Extracts 6 different audio features for robust classification.
