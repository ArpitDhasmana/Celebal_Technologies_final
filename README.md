# 🌿 Plant Condition Classifier

A traditional machine learning–based image classification app that identifies **plant conditions** (e.g., healthy, rust, scab, multiple diseases) from leaf or flower images — using handcrafted features like HOG, LBP, and Color Histograms.

🔍 Powered by `OpenCV`, `scikit-learn`, and `Streamlit`, this lightweight, deployable app avoids deep learning and leverages classic ML techniques for effective image classification.

---

## 🌐 Live Demo

👉 [Click here to try the app online](https://imagecalssification.streamlit.app/)

> ![images (2)](https://github.com/user-attachments/assets/9b4b0043-4479-4448-9c07-78c0d7a6d509)


---

## 🚀 Features

- 🌱 Classifies images into: `healthy`, `rust`, `scab`, `multiple_diseases`
- 🧠 Uses **traditional ML models** (Random Forest)
- 📸 Extracts **handcrafted features** (HOG, LBP, Color Histograms)
- 🎨 Modern frontend built with **Streamlit**
- 💡 Lightweight and easy to deploy

---

## 🖼️ Screenshots

### 🏠 Home Page
> Upload area with a beautiful modern UI

<img width="1919" height="860" alt="Screenshot 2025-07-20 232554" src="https://github.com/user-attachments/assets/d0af058d-2563-46e6-9097-1cb3b27284cc" />


---

### 📤 After Upload
> Shows uploaded image, file name, and prediction result

<img width="1898" height="880" alt="Screenshot 2025-07-20 232719" src="https://github.com/user-attachments/assets/953503fa-edbe-4618-9216-b7cd2c6fca5d" />


---

## 🧑‍💻 Local Setup

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/image_classification_ml
cd image_classification_ml
````

2. **Set up a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install requirements:**

```bash
pip install -r requirements.txt
```

4. **Run the app:**

```bash
streamlit run app.py
```

---

## 🧠 Model & Training

* Features extracted using:

  * 🔷 Histogram of Oriented Gradients (HOG)
  * 🔷 Local Binary Patterns (LBP)
  * 🔷 Color Histograms
* Trained using:

  * 🌲 Random Forest Classifier (via `scikit-learn`)
* Evaluation and training are handled in `train_model.py`

---

## 🧰 Tech Stack

| Tool         | Purpose                     |
| ------------ | --------------------------- |
| Streamlit    | Web frontend                |
| scikit-learn | Model training & prediction |
| OpenCV       | Image processing            |
| Pillow (PIL) | Image I/O and display       |
| NumPy        | Array processing            |
| joblib       | Model serialization         |

---

## 🗂️ Project Structure

```
image_classification/
│
├── app.py                 # Streamlit frontend
├── train_model.py         # Model training script
├── feature_extraction.py  # Handcrafted feature logic
├── utils.py               # Helper functions
├── requirements.txt
├── runtime.txt            # Python version pin
├── models/
│   └── plant_classifier.pkl
├── screenshots/
│   ├── home.png
│   └── prediction.png
└── images/                # Training/test images
```

---

## 🙌 Credits

Built by Arpit Dhasmana
Icons and UI inspired by nature 
