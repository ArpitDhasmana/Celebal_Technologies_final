import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")

from feature_extraction import extract_features

# Config
st.set_page_config(page_title="🌿 Plant Condition Classifier", layout="centered")

# Load Model
model = joblib.load("models/plant_classifier.pkl")

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f4fff9;
    }
    .stButton>button {
        background-color: #00897b;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stFileUploader {
        border: 2px dashed #00897b;
        padding: 1rem;
        border-radius: 8px;
        background-color: #e0f2f1;
    }
    .file-label {
        color: #d0f0f8;
        font-weight: 600;
        font-size: 17px;
        padding-bottom: 8px;
    }
    .file-name {
        color: #d0f0f8;
        font-weight: 500;
        padding-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
st.title("🌿 Plant Condition Classifier")
st.markdown("Upload a plant image (leaf or flower), and the model will predict the plant condition using handcrafted features and traditional machine learning. 💡")

# Custom Label Above File Uploader
st.markdown("<div class='file-label'>📤 Upload Plant Image</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Show filename in soft cyan
    st.markdown(f"<div class='file-name'>📁 {uploaded_file.name}</div>", unsafe_allow_html=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Extract features
    try:
        from feature_extraction import extract_features_from_array
        features = extract_features_from_array(img_bgr).reshape(1, -1)
    except:
        # Fallback if extract_features_from_array doesn't exist
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        features = extract_features(temp_path).reshape(1, -1)
        os.remove(temp_path)

    # Predict
    prediction = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features))

    # Display result
    st.success(f"Predicted Plant Condition: **{prediction}**")
    st.info(f"Confidence: **{confidence:.2%}**")

else:
    st.warning("Please upload a plant image to get started.")

# --- Footer
st.markdown("---")
st.caption("Powered by handcrafted features (HOG, LBP, Color Histograms) and Random Forest. Built with using Streamlit.")
