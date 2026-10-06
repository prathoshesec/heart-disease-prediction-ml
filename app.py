import os
import json
import pickle
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="CardioGuard AI - Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a modern, clinical healthcare aesthetic
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header hero banner */
    .hero-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 50%, #0d9488 100%);
        padding: 26px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(14, 116, 144, 0.2);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 8px;
        line-height: 1.5;
    }

    .badge-container {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .badge {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Result Cards */
    .result-card-high {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 14px;
        padding: 22px;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(244, 63, 94, 0.15);
    }

    .result-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 22px;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.15);
    }

    .risk-title-high {
        color: #be123c;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }

    .risk-title-low {
        color: #15803d;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }

    .risk-score {
        font-size: 2.2rem;
        font-weight: 900;
    }

    .flag-pill-danger {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .flag-pill-normal {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #6ee7b7;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Section divider cards */
    .section-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "models", "heart_disease_bundle.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics_summary.json")

@st.cache_resource
def load_bundle_and_models():
    """Load pre-trained models bundle and metrics."""
    if not os.path.exists(BUNDLE_PATH):
        # Auto-train pipeline if bundle not found
        from ml_engine import run_training_pipeline
        run_training_pipeline()

    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.read_csv(DATA_PATH)
    return bundle, metrics, df

try:
    bundle, metrics, df_dataset = load_bundle_and_models()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# Header Banner
st.markdown("""
<div class="hero-box">
    <div class="hero-title">
        <span>❤️</span> CardioGuard AI — Heart Disease Prediction System
    </div>
    <div class="hero-subtitle">
        Clinical Decision-Support Prototype powered by Multi-Algorithm Machine Learning.
        Evaluates 13 key cardiovascular and hemodynamic biomarkers to predict early heart disease probability.
    </div>
    <div class="badge-container">
        <span class="badge">🎯 Champion Model: Logistic Regression (91.7% Accuracy)</span>
        <span class="badge">📊 Benchmark: UCI Cleveland Heart Disease Dataset</span>
        <span class="badge">⚡ Multi-Model Ensemble Support</span>
        <span class="badge">🔒 Offline & Secure</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_predict, tab_models, tab_eda, tab_glossary, tab_disclaimer = st.tabs([
    "🩺 Patient Risk Predictor",
    "📊 Model Benchmark & Comparison",
    "📈 Dataset Explorer & Visualizations",
    "📚 Clinical Feature Glossary",
    "⚠️ Medical Disclaimer"
])

# ==============================================================================
# TAB 1: PATIENT RISK PREDICTOR
# ==============================================================================
with tab_predict:
    st.markdown("### 📋 Enter Patient Clinical Parameters")
    st.markdown("Fill out the patient's diagnostic profile below or use sample presets for instant demonstration.")

    # Preset Profile Loader
    col_preset1, col_preset2, col_preset3, col_preset_space = st.columns([1.5, 1.5, 1.2, 3])
    
    # Initialize session state for inputs
    defaults = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with col_preset1:
        if st.button("🚨 Load High-Risk Sample", use_container_width=True):
            st.session_state.age = 63
            st.session_state.sex = 1
            st.session_state.cp = 0  # Typical angina
            st.session_state.trestbps = 160
            st.session_state.chol = 295
            st.session_state.fbs = 1
            st.session_state.restecg = 1
            st.session_state.thalach = 108
            st.session_state.exang = 1
            st.session_state.oldpeak = 2.8
            st.session_state.slope = 2
            st.session_state.ca = 2
            st.session_state.thal = 3
            st.rerun()

    with col_preset2:
        if st.button("💚 Load Healthy Sample", use_container_width=True):
            st.session_state.age = 42
            st.session_state.sex = 0
            st.session_state.cp = 1  # Atypical angina
            st.session_state.trestbps = 115
            st.session_state.chol = 185
            st.session_state.fbs = 0
            st.session_state.restecg = 0
            st.session_state.thalach = 172
            st.session_state.exang = 0
            st.session_state.oldpeak = 0.2
            st.session_state.slope = 0
            st.session_state.ca = 0
            st.session_state.thal = 2
            st.rerun()

    with col_preset3:
        if st.button("🔄 Reset Defaults", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.write("")

    # Form Layout in 3 Clinical Groups
    col_group1, col_group2, col_group3 = st.columns(3)

    # ---------------- Group 1: Demographics & Chest Pain ----------------
    with col_group1:
        st.markdown("#### 1️⃣ Patient Demographics")
        age = st.slider("Patient Age (Years)", min_value=20, max_value=85, value=int(st.session_state.age), step=1,
                        help="Patient's chronological age in completed years.")
        
        sex_options = {1: "Male (1)", 0: "Female (0)"}
        sex_val = st.radio("Biological Sex", options=[1, 0], format_func=lambda x: sex_options[x],
                           index=0 if st.session_state.sex == 1 else 1, horizontal=True)

        cp_options = {
            0: "0: Typical Angina (Chest pain on exertion)",
            1: "1: Atypical Angina (Non-classic discomfort)",
            2: "2: Non-Anginal Pain (Likely musculoskeletal)",
            3: "3: Asymptomatic (No chest pain reported)"
        }
        cp_val = st.selectbox(
            "Chest Pain Type (cp)",
            options=[0, 1, 2, 3],
            format_func=lambda x: cp_options[x],
            index=int(st.session_state.cp),
            help="Clinical classification of chest discomfort."
        )

        st.markdown("#### 2️⃣ Fluoroscopy & Thalassemia")
        ca_val = st.selectbox(
            "Number of Major Vessels Colored (ca)",
            options=[0, 1, 2, 3],
            index=int(st.session_state.ca),
            help="Number of major coronary vessels (0-3) highlighted during fluoroscopy."
        )

        thal_options = {
            1: "1: Normal blood flow",
            2: "2: Fixed defect (No blood flow in part of heart)",
            3: "3: Reversible defect (Blood flow observed but not normal)"
        }
        thal_val = st.selectbox(
            "Thalassemia Status (thal)",
            options=[1, 2, 3],
            format_func=lambda x: thal_options[x],
            index=[1, 2, 3].index(int(st.session_state.thal)),
            help="Thallium stress test results."
        )

    # ---------------- Group 2: Vitals & Blood Chemistry ----------------
    with col_group2:
        st.markdown("#### 3️⃣ Vitals & Hemodynamics")
        trestbps = st.slider(
            "Resting Blood Pressure (mm Hg)",
            min_value=80, max_value=210, value=int(st.session_state.trestbps), step=1,
            help="Resting arterial blood pressure measured upon hospital admission."
        )
        if trestbps < 120:
            st.caption("🟢 **BP Status**: Normal (<120 mm Hg)")
        elif 120 <= trestbps < 130:
            st.caption("🟡 **BP Status**: Elevated (120 - 129 mm Hg)")
        elif 130 <= trestbps < 140:
            st.caption("🟠 **BP Status**: Stage 1 Hypertension (130 - 139 mm Hg)")
        else:
            st.caption("🔴 **BP Status**: Stage 2 Hypertension (≥140 mm Hg)")

        chol = st.slider(
            "Serum Cholesterol (mg/dl)",
            min_value=120, max_value=580, value=int(st.session_state.chol), step=2,
            help="Total fasting serum cholesterol level."
        )
        if chol < 200:
            st.caption("🟢 **Cholesterol**: Desirable (<200 mg/dl)")
        elif 200 <= chol < 240:
            st.caption("🟡 **Cholesterol**: Borderline High (200 - 239 mg/dl)")
        else:
            st.caption("🔴 **Cholesterol**: High Risk (≥240 mg/dl)")

        fbs_options = {0: "False (≤ 120 mg/dl - Normal)", 1: "True (> 120 mg/dl - Diabetic / Pre-diabetic)"}
        fbs_val = st.radio(
            "Fasting Blood Sugar > 120 mg/dl (fbs)",
            options=[0, 1],
            format_func=lambda x: fbs_options[x],
            index=0 if st.session_state.fbs == 0 else 1
        )

        restecg_options = {
            0: "0: Normal",
            1: "1: ST-T Wave Abnormality (T-wave inversions)",
            2: "2: Left Ventricular Hypertrophy (Estes' criteria)"
        }
        restecg_val = st.selectbox(
            "Resting ECG Results (restecg)",
            options=[0, 1, 2],
            format_func=lambda x: restecg_options[x],
            index=int(st.session_state.restecg)
        )

    # ---------------- Group 3: Cardiac Stress Test ----------------
    with col_group3:
        st.markdown("#### 4️⃣ Cardiac Stress Test Indicators")
        thalach = st.slider(
            "Maximum Heart Rate Achieved (thalach)",
            min_value=70, max_value=210, value=int(st.session_state.thalach), step=1,
            help="Peak heart rate reached during treadmill exercise stress test."
        )
        age_predicted_max = 220 - age
        hr_percentage = round((thalach / age_predicted_max) * 100, 1)
        st.caption(f"ℹ️ Age-predicted Max HR: **{age_predicted_max} bpm** (Achieved: **{hr_percentage}%**)")

        exang_options = {0: "No (0) - No angina under stress", 1: "Yes (1) - Angina triggered by exercise"}
        exang_val = st.radio(
            "Exercise-Induced Angina (exang)",
            options=[0, 1],
            format_func=lambda x: exang_options[x],
            index=0 if st.session_state.exang == 0 else 1
        )

        oldpeak = st.slider(
            "ST Depression Induced by Exercise (oldpeak)",
            min_value=0.0, max_value=6.5, value=float(st.session_state.oldpeak), step=0.1,
            help="ST depression in millimeters relative to resting baseline ECG."
        )
        if oldpeak > 1.5:
            st.caption("🔴 Significant ST depression detected (Ischemic indicator)")

        slope_options = {
            0: "0: Upsloping (Better cardiac reserve)",
            1: "1: Flat (Equivocal / Moderate indicator)",
            2: "2: Downsloping (High indicator of ischemia)"
        }
        slope_val = st.selectbox(
            "Slope of Peak Exercise ST Segment (slope)",
            options=[0, 1, 2],
            format_func=lambda x: slope_options[x],
            index=int(st.session_state.slope)
        )

    # Algorithm selector
    st.write("---")
    col_algo, col_btn = st.columns([2.5, 1.5])
    with col_algo:
        selected_model_name = st.selectbox(
            "🤖 Select Machine Learning Model for Prediction:",
            options=list(bundle["trained_models"].keys()),
            index=list(bundle["trained_models"].keys()).index(bundle["best_model_name"]),
            help="You can compare the diagnosis across different algorithms!"
        )

    with col_btn:
        st.write("")
        st.write("")
        predict_clicked = st.button("🔮 Analyze Heart Disease Risk", type="primary", use_container_width=True)

    # Processing prediction
    active_clf = bundle["trained_models"][selected_model_name]
    use_scaled = next(item["use_scaled"] for item in bundle["results"] if item["model"] == selected_model_name)
    scaler = bundle["scaler"]

    # Assemble feature vector in exact order
    raw_features = np.array([[
        float(age), float(sex_val), float(cp_val), float(trestbps), float(chol),
        float(fbs_val), float(restecg_val), float(thalach), float(exang_val),
        float(oldpeak), float(slope_val), float(ca_val), float(thal_val)
    ]])

    eval_vector = scaler.transform(raw_features) if use_scaled else raw_features
    probabilities = active_clf.predict_proba(eval_vector)[0]
    risk_prob = float(probabilities[1])
    is_disease_likely = (risk_prob >= 0.50)

    # Display Results Card
    st.write("---")
    st.markdown("### 📊 Diagnostic Risk Assessment Result")

    col_res_banner, col_res_details = st.columns([1.4, 1.6])

    with col_res_banner:
        if is_disease_likely:
            st.markdown(f"""
            <div class="result-card-high">
                <div class="risk-title-high">⚠️ HEART DISEASE IS LIKELY</div>
                <p style="margin-top: 6px; color: #475569; font-weight: 500;">
                    The machine learning model detected an <b>elevated clinical risk</b> of coronary heart disease based on the submitted biomarker pattern.
                </p>
                <div style="margin-top: 14px;">
                    <div style="font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Calculated Risk Probability</div>
                    <div class="risk-score" style="color: #e11d48;">{risk_prob * 100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-low">
                <div class="risk-title-low">✅ HEART DISEASE IS NOT LIKELY</div>
                <p style="margin-top: 6px; color: #475569; font-weight: 500;">
                    The model evaluated the patient profile as <b>low risk</b>. Most vital indicators fall within acceptable physiological thresholds.
                </p>
                <div style="margin-top: 14px;">
                    <div style="font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Calculated Risk Probability</div>
                    <div class="risk-score" style="color: #16a34a;">{risk_prob * 100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.progress(risk_prob)
        st.caption(f"Risk Meter: **{risk_prob * 100:.1f}%** (Algorithm: **{selected_model_name}**)")

    with col_res_details:
        st.markdown("#### 🔎 Key Contributing Clinical Risk Factors")
        
        flags = []
        if trestbps >= 140:
            flags.append(("Stage 2 Hypertension", f"Resting BP {trestbps} mm Hg is high (threshold: 140)", "danger"))
        elif trestbps >= 130:
            flags.append(("Stage 1 Hypertension", f"Resting BP {trestbps} mm Hg is elevated", "danger"))
        else:
            flags.append(("Blood Pressure", f"Normal ({trestbps} mm Hg)", "normal"))

        if chol >= 240:
            flags.append(("Hypercholesterolemia", f"Serum Cholesterol {chol} mg/dl is high (desirable: <200)", "danger"))
        elif chol >= 200:
            flags.append(("Borderline Cholesterol", f"Cholesterol {chol} mg/dl is borderline", "danger"))
        else:
            flags.append(("Cholesterol", f"Optimal ({chol} mg/dl)", "normal"))

        if cp_val == 0:
            flags.append(("Typical Angina", "Significant predictor of myocardial ischemia", "danger"))
        elif cp_val == 1:
            flags.append(("Atypical Angina", "Moderate clinical watchfulness indicated", "danger"))

        if exang_val == 1:
            flags.append(("Exercise Angina", "Chest tightness triggered under physical exertion", "danger"))

        if oldpeak >= 1.5:
            flags.append(("ST Depression", f"Exercise depression {oldpeak} mm indicates ischemia", "danger"))

        if ca_val >= 1:
            flags.append(("Vessel Blockage", f"{ca_val} major coronary vessel(s) detected with fluoroscopy", "danger"))

        if thal_val == 3:
            flags.append(("Reversible Thallium Defect", "Impaired perfusion during cardiac stress", "danger"))

        if fbs_val == 1:
            flags.append(("Hyperglycemia", "Fasting blood sugar > 120 mg/dl is a metabolic risk", "danger"))

        # Render flags
        flag_html = ""
        for tag, detail, status in flags:
            cls = "flag-pill-danger" if status == "danger" else "flag-pill-normal"
            symbol = "⚠️" if status == "danger" else "✓"
            flag_html += f'<div style="margin-bottom: 8px;"><span class="{cls}">{symbol} {tag}</span> <small style="color: #475569;">{detail}</small></div>'
        
        st.markdown(flag_html, unsafe_allow_html=True)

        st.markdown("#### 💡 Clinical Next Steps & Advice")
        if is_disease_likely:
            st.info("""
            1. **Consult a Cardiologist**: Comprehensive formal evaluation (Echocardiogram, Coronary Angiogram, or Cardiac CT).
            2. **Cardiovascular Risk Reduction**: Evaluate statin therapy for lipids, anti-hypertensives, and lifestyle modifications.
            3. **Follow-up Monitoring**: Regular blood pressure and glycemic tracking.
            """)
        else:
            st.success("""
            1. **Maintain Healthy Lifestyle**: Follow a Mediterranean-style or heart-healthy diet low in saturated fats.
            2. **Aerobic Exercise**: Maintain 150 minutes per week of moderate-intensity cardiovascular activity.
            3. **Routine Annual Health Screening**: Continue monitoring lipid panels and blood pressure yearly.
            """)

    # Downloadable Clinical Summary Report
    st.write("---")
    report_text = f"""========================================================================
CARDIOGUARD AI - PATIENT HEART DISEASE RISK ASSESSMENT REPORT
========================================================================
Report Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Evaluated With: {selected_model_name}
Model Accuracy: {next(item['accuracy']*100 for item in bundle['results'] if item['model'] == selected_model_name):.2f}%

1. PATIENT DEMOGRAPHICS & CLINICAL PARAMETERS:
------------------------------------------------------------------------
- Age: {age} years
- Biological Sex: {"Male" if sex_val == 1 else "Female"}
- Chest Pain Type (cp): {cp_options[cp_val]}
- Resting Blood Pressure: {trestbps} mm Hg
- Serum Cholesterol: {chol} mg/dl
- Fasting Blood Sugar > 120 mg/dl: {"Yes" if fbs_val == 1 else "No"}
- Resting Electrocardiogram (ECG): {restecg_options[restecg_val]}
- Max Heart Rate Achieved: {thalach} bpm (Age Predicted Max: {age_predicted_max} bpm)
- Exercise Induced Angina: {"Yes" if exang_val == 1 else "No"}
- ST Depression Induced by Exercise (Oldpeak): {oldpeak} mm
- ST Slope: {slope_options[slope_val]}
- Major Vessels Colored (ca): {ca_val}
- Thalassemia Status (thal): {thal_options[thal_val]}

2. PREDICTION RESULT:
------------------------------------------------------------------------
Result: {"LIKELY TO HAVE HEART DISEASE" if is_disease_likely else "NOT LIKELY TO HAVE HEART DISEASE"}
Estimated Risk Probability: {risk_prob * 100:.2f}%
Classification Threshold: 50.0%

3. CLINICAL SUMMARY & NOTICES:
------------------------------------------------------------------------
Flags Observed:
{chr(10).join(f"- {tag}: {detail}" for tag, detail, _ in flags)}

DISCLAIMER:
This assessment was generated by an educational machine learning decision-support
prototype. It is NOT a clinical diagnosis or medical device. Always consult a
licensed physician or cardiologist for medical evaluation and treatment.
========================================================================
"""
    st.download_button(
        label="📥 Download Patient Assessment Report (TXT)",
        data=report_text,
        file_name=f"heart_risk_assessment_patient_age_{age}.txt",
        mime="text/plain",
        use_container_width=False
    )


# ==============================================================================
# TAB 2: MODEL BENCHMARK & COMPARISON
# ==============================================================================
with tab_models:
    st.markdown("### 📊 Multi-Algorithm Benchmark & Evaluation")
    st.markdown("We trained and compared 5 distinct classification algorithms on the standardized Cleveland clinical dataset using 80/20 train/test split and 5-fold cross-validation.")

    # Results table
    results_df = pd.DataFrame(bundle["results"])[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_accuracy"]]
    results_df.columns = ["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "5-Fold CV Accuracy"]
    
    # Format percentages
    formatted_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "5-Fold CV Accuracy"]:
        formatted_df[col] = (formatted_df[col] * 100).round(2).astype(str) + "%"

    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 🏆 Accuracy & F1-Score Comparison")
        chart_data = pd.DataFrame({
            "Algorithm": results_df["Algorithm"],
            "Accuracy": results_df["Accuracy"] * 100,
            "F1-Score": results_df["F1-Score"] * 100
        }).set_index("Algorithm")
        st.bar_chart(chart_data)

    with col_chart2:
        st.markdown("#### 🌟 ROC-AUC & Precision Comparison")
        chart_data2 = pd.DataFrame({
            "Algorithm": results_df["Algorithm"],
            "ROC-AUC": results_df["ROC-AUC"] * 100,
            "Precision": results_df["Precision"] * 100
        }).set_index("Algorithm")
        st.bar_chart(chart_data2)

    st.write("---")

    # Confusion Matrix & Feature Importance
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown(f"#### 🔍 Confusion Matrix: {bundle['best_model_name']}")
        cm = bundle["confusion_matrices"][bundle["best_model_name"]]
        tn, fp = cm[0]
        fn, tp = cm[1]

        cm_table = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            columns=["Predicted: Healthy (0)", "Predicted: Disease (1)"],
            index=["Actual: Healthy (0)", "Actual: Disease (1)"]
        )
        st.table(cm_table)
        st.caption(f"**True Negatives (TN):** {tn} | **False Positives (FP):** {fp} | **False Negatives (FN):** {fn} | **True Positives (TP):** {tp}")

    with col_fi:
        st.markdown("#### 📈 Feature Importance Ranking")
        st.markdown("Relative predictive importance of clinical attributes determined by the model:")
        fi_df = pd.DataFrame(list(bundle["feature_importance"].items()), columns=["Feature", "Relative Importance"])
        fi_df["Feature Label"] = fi_df["Feature"].map(bundle["feature_labels"])
        fi_chart = fi_df.set_index("Feature Label")[["Relative Importance"]]
        st.bar_chart(fi_chart, horizontal=True)


# ==============================================================================
# TAB 3: DATASET EXPLORER & EDA
# ==============================================================================
with tab_eda:
    st.markdown("### 📈 Cleveland Clinical Dataset Explorer")
    st.markdown("Explore the underlying training dataset of clinical attributes and their epidemiological patterns.")

    # High-level metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    total_records = len(df_dataset)
    disease_cases = int(df_dataset["target"].sum())
    healthy_cases = total_records - disease_cases
    disease_pct = round((disease_cases / total_records) * 100, 1)

    col_m1.metric("Total Patients", f"{total_records}")
    col_m2.metric("Diagnosed Heart Disease", f"{disease_cases} ({disease_pct}%)")
    col_m3.metric("Healthy / Low Risk", f"{healthy_cases} ({100 - disease_pct}%)")
    col_m4.metric("Clinical Features", f"{len(bundle['feature_cols'])}")

    st.write("")

    # Filter controls
    col_filter_gender, col_filter_target = st.columns(2)
    with col_filter_gender:
        filter_sex = st.multiselect("Filter by Sex:", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", default=[1, 0])
    with col_filter_target:
        filter_target = st.multiselect("Filter by Diagnosis:", options=[1, 0], format_func=lambda x: "Disease Present (1)" if x == 1 else "Healthy (0)", default=[1, 0])

    filtered_df = df_dataset[df_dataset["sex"].isin(filter_sex) & df_dataset["target"].isin(filter_target)]
    st.dataframe(filtered_df.head(25), use_container_width=True)
    st.caption(f"Showing 25 of {len(filtered_df)} matching patient records.")

    st.write("---")
    st.markdown("#### 📊 Exploratory Visual Insights")

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.markdown("**Age Distribution across Risk Groups**")
        age_bins = pd.cut(df_dataset["age"], bins=[25, 45, 55, 65, 80], labels=["<45", "45-54", "55-64", "65+"])
        age_cross = pd.crosstab(age_bins, df_dataset["target"]).rename(columns={0: "Healthy", 1: "Heart Disease"})
        st.bar_chart(age_cross)

    with col_eda2:
        st.markdown("**Heart Disease Incidence by Chest Pain Type**")
        cp_cross = pd.crosstab(df_dataset["cp"], df_dataset["target"]).rename(
            index={0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal", 3: "Asymptomatic"},
            columns={0: "Healthy", 1: "Heart Disease"}
        )
        st.bar_chart(cp_cross)


# ==============================================================================
# TAB 4: CLINICAL FEATURE GLOSSARY
# ==============================================================================
with tab_glossary:
    st.markdown("### 📚 Clinical Feature Reference & Medical Definitions")
    st.markdown("Detailed clinical rationale for each of the 13 diagnostic parameters evaluated by the system:")

    glossary = [
        ("1. Age (`age`)", "Chronological age in years. Advanced age is one of the strongest unmodifiable risk factors for atherosclerotic cardiovascular disease."),
        ("2. Sex (`sex`)", "Biological sex (1 = Male, 0 = Female). Men generally face higher early incidence of coronary artery disease; post-menopausal women face rising cardiovascular risk."),
        ("3. Chest Pain Type (`cp`)", "Categorized into four types: 0 = Typical Angina (substernal discomfort brought on by exertion), 1 = Atypical Angina, 2 = Non-Anginal pain, 3 = Asymptomatic."),
        ("4. Resting Blood Pressure (`trestbps`)", "Arterial pressure in mm Hg measured at rest. Chronic hypertension (≥130/80 mm Hg) damages endothelial lining, accelerating plaque formation."),
        ("5. Serum Cholesterol (`chol`)", "Total serum cholesterol in mg/dL. Elevated low-density lipoprotein (LDL) and total cholesterol contribute directly to arterial plaque buildup."),
        ("6. Fasting Blood Sugar (`fbs`)", "Indicates whether fasting glucose exceeds 120 mg/dL (1 = True, 0 = False). Diabetes mellitus is a major multiplier of coronary disease risk."),
        ("7. Resting Electrocardiogram (`restecg`)", "Baseline electrical conduction of the heart: 0 = Normal, 1 = ST-T wave abnormalities (indicates ischemia), 2 = Left Ventricular Hypertrophy (LVH)."),
        ("8. Maximum Heart Rate Achieved (`thalach`)", "Peak heart rate reached during graded treadmill exercise stress testing. Reduced maximal heart rate indicates chronotropic incompetence and diminished cardiovascular reserve."),
        ("9. Exercise-Induced Angina (`exang`)", "Occurrence of chest discomfort provoked by physical stress (1 = Yes, 0 = No). Strong indicator of flow-limiting coronary stenoses."),
        ("10. ST Depression (`oldpeak`)", "Magnitude of ST-segment depression in millimeters induced by exercise relative to rest. Reflects subendocardial myocardial ischemia."),
        ("11. Slope of Peak ST Segment (`slope`)", "The trajectory of the ST segment at peak exertion: 0 = Upsloping, 1 = Flat, 2 = Downsloping (downsloping has the highest correlation with multi-vessel disease)."),
        ("12. Major Vessels Colored (`ca`)", "Number of major coronary arteries (0 - 3) visualized with contrast during fluoroscopy. Higher numbers signify extensive arterial obstruction."),
        ("13. Thalassemia Status (`thal`)", "Nuclear scintigraphy perfusion defect: 1 = Normal perfusion, 2 = Fixed defect (non-viable scar tissue from prior infarction), 3 = Reversible defect (viable but ischemic tissue).")
    ]

    for title, desc in glossary:
        with st.expander(title, expanded=False):
            st.write(desc)


# ==============================================================================
# TAB 5: MEDICAL DISCLAIMER
# ==============================================================================
with tab_disclaimer:
    st.markdown("### ⚠️ Formal Medical & Educational Prototype Notice")
    st.warning("""
    **EDUCATIONAL AND RESEARCH PROTOTYPE ONLY**
    
    This application is an academic software demonstration designed to showcase the application of Machine Learning algorithms (Logistic Regression, Random Forest, Decision Trees, KNN, and Naive Bayes) in cardiovascular risk assessment.

    **Critical Points for Users & Evaluators:**
    1. **Not a Medical Device**: This software has NOT been approved or cleared by any national or international healthcare authority (such as the US FDA, EMA, or CDSCO).
    2. **Not a Substitute for Professional Consultation**: The risk estimates provided by this prototype must NOT be used as the sole basis for diagnosis, medical treatment, prescription, or clinical decision-making.
    3. **Consult a Doctor**: If you or anyone you know experiences chest pressure, shortness of breath, radiating arm or jaw pain, dizziness, or irregular heartbeat, seek emergency medical care or consult a certified physician immediately.
    4. **Clinical Validation Required**: Real-world deployment requires extensive multi-center clinical trials, privacy governance (HIPAA/GDPR), and integration with Electronic Health Record (EHR) systems.
    """)

# Footer
st.write("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 10px 0;">
    CardioGuard AI — Heart Disease Prediction System | Built with Python, Streamlit, NumPy & Pandas | Educational Prototype
</div>
""", unsafe_allow_html=True)
