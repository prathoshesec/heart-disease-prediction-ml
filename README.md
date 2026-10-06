 ❤️ HEART DISEASE PREDICTION USING MACHINE LEARNING

A Machine Learning-powered Heart Disease Prediction System that analyzes patient health-related features, compares multiple machine learning algorithms, and predicts the presence of heart disease through an interactive Streamlit dashboard.

Live Website | Python | Streamlit | scikit-learn**

🌐 Live Demo:  
[Heart Disease Prediction – Live App](https://heart-disease-prediction-ml-h5rjbmyam5tkers3httcp.streamlit.app?utm_source=chatgpt.com)

---

🌟 KEY HIGHLIGHTS & FEATURES

1. 🩺 Interactive Heart Disease Prediction

Smart Inputs:
Enter patient-related health parameters such as age, sex, chest pain type, blood pressure, cholesterol, maximum heart rate, and other clinical features.

Instant Prediction:
The system analyzes the entered values and provides a predicted risk category.

Model Selection: 
Users can select from the available trained machine learning models.

---

2. 🤖 Machine Learning Model Dashboard

Multiple Algorithms: 
The project compares multiple machine learning algorithms:

- Random Forest
- Logistic Regression
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

Champion Model: 
🏆 Support Vector Machine (SVM)

Best Accuracy:  
🎯 85.00%

The SVM model achieved the best evaluation result among the models tested in this project.

---

3. 📊 Model Performance Comparison

The application provides a comparison of the trained models.

| Model | Accuracy |
|---|---:|
| 🏆 SVM | 85.00% |
| Logistic Regression | 83.33% |
| KNN | 83.33% |
| Random Forest | 81.67% |
| Gradient Boosting | 81.67% |
| Decision Tree | 75.00% |

---

4. 📈 Dataset Explorer & Analytics

Dataset Information:

- Dataset: UCI Cleveland Heart Disease Dataset
- Raw records: 303
- Complete records used: 297
- Features: 13
- Classification: Binary

The dashboard provides dataset information and allows users to explore the available patient features and target distribution.

---

5. 🧪 Sample Prediction Testing

The application can be tested using different patient input combinations.

High-Risk Sample: 
Used to demonstrate a higher predicted risk result.

Healthy Sample:  
Used to demonstrate a lower predicted risk result.

Reset Option:  
Allows the user to return the prediction inputs to their default values.

---

📁 PROJECT DIRECTORY STRUCTURE

```text
heart-disease-prediction-ml/
│
├── app.py
├── heart.csv
├── heart_disease_bundle.joblib
├── metrics_summary.json
├── heart_disease_bundle.pkl
├── requirements.txt
├── README.md
│
└── models/
    └── heart_disease_bundle.joblib
```

> Note: The exact files/folders in your GitHub repository may vary depending on your final upload structure.

---

🚀 QUICK START GUIDE

1. 🌐 Launching the Live Application

Open the deployed Streamlit application:

[Heart Disease Prediction – Live Demo](https://heart-disease-prediction-ml-h5rjbmyam5tkers3httcp.streamlit.app?utm_source=chatgpt.com)

The application provides an interactive interface for entering patient information and generating predictions.

---

## 2. 💻 Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/prathoshesec/heart-disease-prediction-ml.git
```

Move into the project folder:

```bash
cd heart-disease-prediction-ml
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

3. 📝 Testing the Prediction Workflow

1. Open the application.
2. Enter the patient-related input values.
3. Select a machine learning model.
4. Click Analyze.
5. View the predicted risk result.
6. Compare model performance using the Model Benchmark section.
7. Explore the dataset using Dataset Explorer.

---

🧠 MODEL INPUT & OUTPUT QUICK REFERENCE

💻 This is a "software-only machine learning project", so no physical hardware or wiring is required.

| Type | Feature | Description |
|---|---|---|
| Input | Age | Patient age |
| Input | Sex | Patient sex |
| Input | Chest Pain | Type of chest pain |
| Input | Resting Blood Pressure | Resting blood pressure |
| Input | Cholesterol | Cholesterol level |
| Input | Fasting Blood Sugar | Fasting blood sugar status |
| Input | Resting ECG | Resting electrocardiographic result |
| Input | Maximum Heart Rate | Maximum heart rate achieved |
| Input | Exercise Angina | Exercise-induced angina |
| Input | Oldpeak | ST depression |
| Input | Slope | ST segment slope |
| Input | CA | Number of major vessels |
| Input | Thal | Thalassemia-related value |
| Output | Predicted Risk | Higher or Lower Predicted Risk |

---

📊 DATASET

UCI Cleveland Heart Disease Dataset

The project uses the UCI Cleveland Heart Disease Dataset.

Dataset statistics:

- Raw instances: 303
- Complete records used: 297
- No Disease: 160
- Disease: 137
- Features: 13
- Target: Binary classification

The original heart disease target values are converted into a binary classification for prediction.

---

🤖 MACHINE LEARNING WORKFLOW

```text
UCI Cleveland Dataset
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Data Preprocessing
        ↓
Train Multiple ML Models
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
SVM Selected as Champion Model
        ↓
Streamlit Prediction Dashboard
        ↓
Heart Disease Risk Prediction
```

---

🏆 CHAMPION MODEL

### Support Vector Machine — SVM

Accuracy: 85.00%

SVM achieved the best evaluation result among the machine learning algorithms tested in this project.

The model is used to classify the input into the corresponding predicted risk category.

---

🛠️ TECH STACK

| Layer | Technology |
|---|---|
| Programming | Python |
| Machine Learning | scikit-learn |
| Data Processing | pandas, NumPy |
| Model Storage | Joblib |
| Dashboard | Streamlit |
| Dataset | UCI Cleveland Heart Disease Dataset |
| Version Control | Git & GitHub |

---

📌 PROJECT OBJECTIVES

- To develop a machine learning-based heart disease prediction system.
- To use a real-world heart disease dataset.
- To preprocess and analyze patient-related features.
- To train multiple machine learning algorithms.
- To compare model performance.
- To select the best-performing model.
- To deploy the prediction system as an interactive Streamlit application.

---

💡 PROJECT BENEFITS

- Easy-to-use prediction interface.
- Multiple machine learning models for comparison.
- Real-world dataset usage.
- Interactive Streamlit dashboard.
- Fast prediction from user-provided inputs.
- Useful for demonstrating machine learning classification concepts.

---

⚠️ DISCLAIMER

This project is developed for **educational and machine learning demonstration purposes only**.

It is **not a medical device** and should **not be used as a substitute for professional medical diagnosis or treatment**.

The predictions generated by this application are model-based outputs and should not be interpreted as medical advice.

---

📄 PROJECT LINKS

🌐 Live Application:
[Heart Disease Prediction – Streamlit App](https://heart-disease-prediction-ml-h5rjbmyam5tkers3httcp.streamlit.app?utm_source=chatgpt.com)

💻 GitHub Repository:  
[Heart Disease Prediction – GitHub Repository](https://github.com/prathoshesec/heart-disease-prediction-ml?utm_source=chatgpt.com)

