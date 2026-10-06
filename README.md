# HEART DISEASE PREDICTION USING MACHINE LEARNING ❤️🤖
A Machine Learning-Powered Heart Disease Risk Prediction System with Model Comparison, Dataset Exploration, and an Interactive Streamlit Dashboard.

[![Live App](https://img.shields.io/badge/Live-App-brightgreen)](https://heart-disease-prediction-ml-h5rjbmyam5tkers3httcp.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)

🌐 Live Demo: https://heart-disease-prediction-ml-h5rjbmyam5tkers3httcp.streamlit.app

## 🌟 KEY HIGHLIGHTS & FEATURES

### 1. 🩺 Interactive Heart Disease Prediction
- **Smart Inputs:** Enter patient health parameters such as age, sex, chest pain type, blood pressure, cholesterol, maximum heart rate, and other clinical features.
- **Instant Results:** View the predicted risk category (Higher or Lower Predicted Risk).
- **Model Selection:** Choose from the available trained machine learning models.

### 2. 🤖 Machine Learning Model Dashboard
- **Multiple Algorithms:** Random Forest, Logistic Regression, Gradient Boosting, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Decision Tree.
- **Champion Model:** 🏆 Support Vector Machine (SVM).
- **Best Accuracy:** 🎯 85.00% on the test data.

### 3. 📊 Model Performance Comparison
- **Benchmark View:** Compare the accuracy of all trained models side by side.

| Model | Accuracy |
|---|---:|
| 🏆 SVM | 85.00% |
| Logistic Regression | 83.33% |
| KNN | 83.33% |
| Random Forest | 81.67% |
| Gradient Boosting | 81.67% |
| Decision Tree | 75.00% |

### 4. 📈 Dataset Explorer & Analytics
- **Dataset Information:** UCI Cleveland Heart Disease Dataset with 303 raw records, 297 complete records, and 13 features.
- **Target Distribution:** Explore patient features and the disease / no-disease split.

### 5. 🧪 Sample Prediction Testing
- **High-Risk Sample:** Try a sample that demonstrates a higher predicted risk.
- **Healthy Sample:** Try a sample that demonstrates a lower predicted risk.
- **Reset Option:** Return all inputs to their default values.

## 📁 PROJECT DIRECTORY STRUCTURE
```text
heart-disease-prediction-ml/
├── app.py                        # Streamlit dashboard
├── heart.csv                     # UCI Cleveland dataset
├── heart_disease_bundle.joblib   # Trained model bundle
├── heart_disease_bundle.pkl      # Trained model bundle (pickle)
├── metrics_summary.json          # Model evaluation metrics
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── models/
    └── heart_disease_bundle.joblib
```
📝 Note: The exact files and folders may vary depending on your final upload structure.

## 🚀 QUICK START GUIDE (HOW TO RUN)

### 1. Launching the Live Application
Open the deployed Streamlit app directly: Heart Disease Prediction Live Demo

### 2. Running the Project Locally
Clone the repository:
```bash
git clone https://github.com/prathoshesec/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```
Start the dashboard:
```bash
streamlit run app.py
```
Then open your browser at http://localhost:8501.

### 3. Testing the Workflow Instantly
1. Open the Live App or the local Streamlit dashboard.
2. Enter the patient-related input values.
3. Select a machine learning model.
4. Click Analyze and view the predicted risk result.
5. Compare model performance in the Model Benchmark section.
6. Explore the data using Dataset Explorer.

## 🧠 MODEL INPUT & OUTPUT QUICK REFERENCE
💻 This is a software-only project, so no physical hardware or wiring is required.

| Type | Details | Notes |
|---|---|---|
| Input | Age | Patient age |
| Input | Sex | Patient sex |
| Input | Chest Pain | Type of chest pain |
| Input | Resting Blood Pressure | Resting BP value |
| Input | Cholesterol | Cholesterol level |
| Input | Fasting Blood Sugar | Fasting blood sugar status |
| Input | Resting ECG | Resting electrocardiographic result |
| Input | Maximum Heart Rate | Max heart rate achieved |
| Input | Exercise Angina | Exercise-induced angina |
| Input | Oldpeak | ST depression |
| Input | Slope | ST segment slope |
| Input | CA | Number of major vessels |
| Input | Thal | Thalassemia-related value |
| Output | Predicted Risk | Higher or Lower Predicted Risk |

## 🤖 MACHINE LEARNING WORKFLOW
```text
UCI Cleveland Dataset
        ↓
Data Cleaning → Feature Selection → Preprocessing
        ↓
Train Multiple ML Models
        ↓
Model Evaluation & Comparison
        ↓
SVM Selected as Champion Model
        ↓
Streamlit Prediction Dashboard
```

## 📊 DATASET DETAILS
| Detail | Value |
|---|---|
| Source | UCI Cleveland Heart Disease Dataset |
| Raw Instances | 303 |
| Complete Records Used | 297 |
| No Disease | 160 |
| Disease | 137 |
| Features | 13 |
| Target | Binary Classification |

## 📄 DOCUMENTATION LINKS
- 🌐 Live Application: https://heart-disease-prediction-ml-h5rjbmyam5tkers3httcp.streamlit.app
- 💻 GitHub Repository: https://github.com/prathoshesec/heart-disease-prediction-ml
- 🖥️ Streamlit Dashboard: app.py
- 📚 Dataset: heart.csv
- 📦 Python Dependencies: requirements.txt

## 🛠️ TECH STACK
| Layer | Technology |
|---|---|
| Programming | Python |
| Machine Learning | scikit-learn |
| Data Processing | pandas, NumPy |
| Model Storage | Joblib |
| Dashboard | Streamlit |
| Dataset | UCI Cleveland Heart Disease Dataset |
| Version Control | Git & GitHub |

## ⚠️ DISCLAIMER & ATTRIBUTION
This project is developed for educational and machine learning demonstration purposes only.

It is not a medical device and should not be used as a substitute for professional medical diagnosis or treatment. Predictions are model-based outputs and should not be interpreted as medical advice.

The project uses the publicly available UCI Cleveland Heart Disease Dataset. No license file is currently included in this repository. Add one if you want to define how others may reuse or modify the project.

## 👨‍💻 DEVELOPED BY
Prathosh S, B.Tech – Artificial Intelligence and Data Science

Developed with ❤️ for smarter, data-driven healthcare awareness.
