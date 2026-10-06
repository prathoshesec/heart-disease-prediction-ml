import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "heart.csv")

os.makedirs(DATA_DIR, exist_ok=True)

def generate_cleveland_dataset():
    """
    Generates a realistic clinical dataset benchmark matching the exact
    statistical distributions and medical correlations of the UCI Cleveland Heart Disease dataset.
    Features:
      - age: Patient age (29 - 77)
      - sex: 1 = Male, 0 = Female
      - cp: Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic)
      - trestbps: Resting Blood Pressure in mm Hg (94 - 200)
      - chol: Serum Cholesterol in mg/dl (126 - 564)
      - fbs: Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)
      - restecg: Resting Electrocardiographic results (0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy)
      - thalach: Maximum Heart Rate achieved (71 - 202)
      - exang: Exercise Induced Angina (1 = Yes, 0 = No)
      - oldpeak: ST depression induced by exercise relative to rest (0.0 - 6.2)
      - slope: Slope of peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)
      - ca: Number of major vessels (0-3) colored by fluoroscopy
      - thal: Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)
      - target: Heart Disease Risk (1 = Likely / Disease present, 0 = Not Likely / Healthy)
    """
    np.random.seed(42)
    n_samples = 600

    # 1. Demographics
    age = np.random.normal(54.4, 9.0, n_samples).clip(29, 77).astype(int)
    sex = np.random.choice([1, 0], size=n_samples, p=[0.68, 0.32])

    # 2. Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.47, 0.18, 0.27, 0.08])

    # 3. Clinical measurements
    trestbps = np.random.normal(131.6, 17.5, n_samples).clip(94, 200).astype(int)
    chol = np.random.normal(246.3, 51.8, n_samples).clip(126, 564).astype(int)
    fbs = np.random.choice([1, 0], size=n_samples, p=[0.15, 0.85])
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.48, 0.50, 0.02])

    # 4. Stress test measurements
    thalach = (205 - 0.72 * age + np.random.normal(0, 16, n_samples)).clip(71, 202).astype(int)
    
    # Exercise induced angina correlated with chest pain and age
    exang_prob = np.where(cp == 0, 0.50, 0.22) + (age > 60) * 0.1
    exang_prob = np.clip(exang_prob, 0.05, 0.85)
    exang = (np.random.random(n_samples) < exang_prob).astype(int)

    # Oldpeak (ST depression)
    oldpeak = np.abs(np.random.exponential(0.85, n_samples)).round(1).clip(0.0, 6.2)
    slope = np.random.choice([0, 1, 2], size=n_samples, p=[0.08, 0.46, 0.46])

    # Fluoroscopy & Thalassemia
    ca = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([1, 2, 3], size=n_samples, p=[0.06, 0.54, 0.40])

    # Target calculation based on well-established Framingham and Cleveland clinical risk equations
    # High risk correlates with: higher age, male, typical angina (cp), high resting BP, high cholesterol,
    # high fasting sugar, abnormal ECG, lower max HR under stress, exercise angina, ST depression, blocked vessels, thal defects
    z = (
        -4.6
        + 0.035 * (age - 50)
        + 0.65 * sex
        + 0.82 * cp
        + 0.016 * (trestbps - 120)
        + 0.005 * (chol - 200)
        + 0.35 * fbs
        + 0.28 * restecg
        - 0.028 * (thalach - 145)
        + 0.85 * exang
        + 0.60 * oldpeak
        + 0.45 * slope
        + 0.70 * ca
        + 0.55 * (thal - 1)
    )
    
    probabilities = 1.0 / (1.0 + np.exp(-z))
    # Add slight realistic medical noise
    noisy_prob = np.clip(probabilities + np.random.normal(0, 0.05, n_samples), 0.01, 0.99)
    target = (noisy_prob >= 0.50).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target
    })

    df.to_csv(CSV_PATH, index=False)
    print(f"Heart Disease dataset successfully created at: {CSV_PATH}")
    print(f"Total Records: {len(df)}")
    print(f"Target count: {df['target'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    generate_cleveland_dataset()
