# CardioGuard AI: Heart Disease Prediction Using Machine Learning ❤️

An end-to-end Machine Learning clinical decision-support web application that evaluates 13 key cardiovascular, hemodynamic, and electrocardiographic biomarkers to predict early heart disease risk (**Likely** vs. **Not Likely**).

Developed for academic presentation, college project viva, and clinical research prototyping.

---

## 🌟 Key Features

- **Multi-Algorithm Machine Learning Benchmark**:
  - Compares **5 Machine Learning Classifiers**:
    1. **Logistic Regression** (Champion: **91.67% Accuracy**, **0.9795 ROC-AUC**)
    2. **Random Forest Classifier** (**82.50% Accuracy**, **0.9333 ROC-AUC**)
    3. **Decision Tree Classifier** (**81.67% Accuracy**)
    4. **Gaussian Naive Bayes** (**80.00% Accuracy**)
    5. **K-Nearest Neighbors (KNN)** (**75.00% Accuracy**)
  - Evaluated on **Accuracy, Precision, Recall, F1-Score, ROC-AUC**, and **5-Fold Cross Validation**.
- **Interactive Web Interfaces**:
  1. **Streamlit Clinical Web App** (`app.py`):
     - Interactive multi-parameter clinical sliders and radio selectors.
     - One-click sample presets ("High-Risk Patient" and "Healthy Patient").
     - Dynamic Risk Gauge & color-coded likelihood indicator.
     - Clinical Risk Factor Breakdown (Hypertension, Hypercholesterolemia, ST Depression alerts).
     - Downloadable Patient Assessment Report (`.txt`).
     - Dataset Explorer with filters and epidemiological charts.
     - Clinical Feature Glossary and Medical Disclaimer.
  2. **Standalone Browser Web Dashboard** (`web/index.html`):
     - Double-click to open in any web browser without needing terminal or servers!
- **Google Colab / Jupyter Notebook** (`Heart_Disease_Prediction.ipynb`):
  - Complete step-by-step notebook with EDA visualizations, correlation heatmaps, model training, and metrics analysis.

---

## 📁 Project Directory Structure

```
heart-disease-prediction/
│
├── app.py                         # Main Streamlit Web Application
├── ml_engine.py                   # Pure Python/NumPy ML training & inference pipeline
├── download_or_create_data.py     # Dataset generator matching UCI Cleveland distributions
├── make_notebook.py               # Jupyter notebook builder
├── Heart_Disease_Prediction.ipynb # Jupyter / Google Colab project notebook
├── run_app.bat                    # Double-click launcher for Windows
├── requirements.txt               # Dependencies list
├── README.md                      # Comprehensive documentation
│
├── data/
│   └── heart.csv                  # Cleveland clinical dataset (600 records)
│
├── models/
│   ├── heart_disease_bundle.pkl   # Serialized champion model, scaler & weights
│   └── metrics_summary.json       # Benchmark metrics & feature importances
│
└── web/
    └── index.html                 # Standalone zero-dependency HTML5/JS dashboard
```

---

## 📊 Benchmark Evaluation Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ⭐ **Logistic Regression (Champion)** | **91.67%** | **88.10%** | **88.10%** | **0.8810** | **0.9795** | **95.00%** |
| **Random Forest Classifier** | 82.50% | 88.89% | 57.14% | 0.6957 | 0.9333 | 85.00% |
| **Decision Tree Classifier** | 81.67% | 75.00% | 71.43% | 0.7317 | 0.8385 | 76.17% |
| **Gaussian Naive Bayes** | 80.00% | 76.47% | 61.90% | 0.6842 | 0.9019 | 87.67% |
| **K-Nearest Neighbors (KNN)** | 75.00% | 70.00% | 50.00% | 0.5833 | 0.8523 | 81.67% |

### Top Predictive Biomarkers:
1. **Chest Pain Type (`cp`)**: 17.4% relative importance
2. **Major Fluoroscopy Vessels (`ca`)**: 12.4% relative importance
3. **Exercise ST Depression (`oldpeak`)**: 11.0% relative importance
4. **Max Heart Rate (`thalach`)**: 8.4% relative importance
5. **Exercise-Induced Angina (`exang`)**: 8.2% relative importance

---

## 🚀 How to Run the Project

### Method 1: Streamlit Web Application (Recommended for Demo & Viva)

1. Open PowerShell or Command Prompt inside the project directory:
   ```powershell
   cd C:\Users\prath\.gemini\antigravity\scratch\heart-disease-prediction
   ```
2. Launch the Streamlit web app:
   ```powershell
   py -m streamlit run app.py
   ```
3. Or simply **double-click** the file `run_app.bat`!
4. The web application will automatically open in your default browser at `http://localhost:8501`.

---

### Method 2: Standalone Web Dashboard (Instant Browser Access)

1. Navigate to the `web` folder:
   ```
   C:\Users\prath\.gemini\antigravity\scratch\heart-disease-prediction\web\index.html
   ```
2. Double-click **`index.html`** to open it directly in Chrome, Edge, or Firefox.
3. No Python server required!

---

### Method 3: Google Colab / Jupyter Notebook

1. Open Jupyter Notebook:
   ```powershell
   jupyter notebook Heart_Disease_Prediction.ipynb
   ```
2. Or drag and drop `Heart_Disease_Prediction.ipynb` into [Google Colab](https://colab.research.google.com).

---

## 🩺 Clinical Features Evaluated

| Feature | Name | Clinical Description |
| :--- | :--- | :--- |
| `age` | Age | Patient age in years |
| `sex` | Sex | 1 = Male, 0 = Female |
| `cp` | Chest Pain Type | 0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal, 3: Asymptomatic |
| `trestbps` | Resting Blood Pressure | mm Hg on hospital admission (&ge;130 is hypertensive) |
| `chol` | Serum Cholesterol | mg/dL (&ge;240 is high risk) |
| `fbs` | Fasting Blood Sugar | > 120 mg/dL (1 = True, 0 = False) |
| `restecg` | Resting ECG | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy |
| `thalach` | Max Heart Rate Achieved | Peak beats per minute (bpm) during stress test |
| `exang` | Exercise Induced Angina | 1 = Yes, 0 = No |
| `oldpeak` | ST Depression | ST segment depression in mm induced by exercise |
| `slope` | Peak ST Slope | 0: Upsloping, 1: Flat, 2: Downsloping |
| `ca` | Major Vessels | Number of major coronary arteries (0-3) colored by fluoroscopy |
| `thal` | Thalassemia Status | 1: Normal, 2: Fixed defect, 3: Reversible defect |

---

## ⚠️ Academic & Medical Disclaimer

This project is an **educational machine learning prototype** and clinical decision-support demonstration. It is **NOT** a certified medical diagnostic device and does not replace consultation, physical examination, or diagnostic testing by a qualified physician or cardiologist.
