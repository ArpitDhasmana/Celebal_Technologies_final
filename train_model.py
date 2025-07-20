import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from feature_extraction import extract_features
from utils import load_dataset

# --- Paths
CSV_PATH = "train.csv"
IMAGE_FOLDER = "images/"
MODEL_PATH = "models/plant_classifier.pkl"

# --- Load dataset using utils.py
image_paths, labels = load_dataset(CSV_PATH, IMAGE_FOLDER, include_labels=True)

X = []
y = []

# --- Feature Extraction
print("🔍 Extracting features from images...")
for path, label in zip(image_paths, labels):
    try:
        features = extract_features(path)
        X.append(features)
        y.append(label)
    except Exception as e:
        print(f"⚠️ Skipping {path}: {e}")

X = np.array(X)
y = np.array(y)
print(f"✅ Feature extraction complete. Total samples: {len(X)}")

# --- Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Model Training
print("🧠 Training Random Forest classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# --- Save Model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
