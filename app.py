import os
import json
import datetime
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CardioGuard AI - Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

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
    background: rgba(255,255,255,0.2);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.3);
}

.result-card-high {
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 2px solid #f43f5e;
    border-radius: 14px;
    padding: 22px;
    margin-top: 15px;
}

.result-card-low {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 22px;
    margin-top: 15px;
}

.risk-title-high {
    color: #be123c;
    font-size: 1.6rem;
    font-weight: 800;
}

.risk-title-low {
    color: #15803d;
    font-size: 1.6rem;
    font-weight: 800;
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "heart_disease_bundle.joblib"
)

METRICS_PATH = os.path.join(
    BASE_DIR, "models", "metrics_summary.json"
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "heart.csv"
)

# Fallback for your current GitHub root files
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(
        BASE_DIR, "heart_disease_bundle.joblib"
    )

if not os.path.exists(METRICS_PATH):
    METRICS_PATH = os.path.join(
        BASE_DIR, "metrics_summary.json"
    )

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(
        BASE_DIR, "heart.csv"
    )

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_system():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "heart_disease_bundle.joblib was not found."
        )

    bundle = joblib.load(MODEL_PATH)

    metrics = None

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    dataset = None

    if os.path.exists(DATA_PATH):
        dataset = pd.read_csv(DATA_PATH)

    return bundle, metrics, dataset


try:
    bundle, metrics, df_dataset = load_system()

except Exception as e:
    st.error("❌ Error loading model files.")
    st.code(str(e))
    st.stop()

# ============================================================
# GET MODEL INFORMATION
# ============================================================
best_model_name = bundle.get("best_model_name")

best_model = bundle.get("best_model")

scaler = bundle.get("scaler")

is_scaled = bundle.get("is_scaled", False)

feature_cols = bundle.get(
    "feature_cols",
    [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]
)

results = bundle.get("results", [])

# New training bundle may contain all_models
all_models = bundle.get("all_models", {})

# Compatibility with older bundle
if not all_models:
    all_models = bundle.get("trained_models", {})

# ============================================================
# FIND BEST MODEL ACCURACY
# ============================================================
best_accuracy = None

for item in results:
    if item.get("model") == best_model_name:
        best_accuracy = item.get("accuracy")
        break

if best_accuracy is None:
    best_accuracy = 0

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="hero-box">

<div class="hero-title">
❤️ CardioGuard AI — Heart Disease Prediction System
</div>

<div class="hero-subtitle">
Machine Learning based heart disease risk prediction using
clinical parameters from the UCI Cleveland Heart Disease dataset.
</div>

<div class="badge-container">

<span class="badge">
🎯 Champion Model: {best_model_name}
</span>

<span class="badge">
📊 Accuracy: {best_accuracy * 100:.2f}%
</span>

<span class="badge">
📚 UCI Cleveland Dataset
</span>

<span class="badge">
⚡ Machine Learning
</span>

</div>

</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab_predict, tab_models, tab_eda, tab_glossary, tab_disclaimer = st.tabs(
    [
        "🩺 Patient Risk Predictor",
        "📊 Model Benchmark",
        "📈 Dataset Explorer",
        "📚 Feature Glossary",
        "⚠️ Disclaimer"
    ]
)

# ============================================================
# TAB 1 - PREDICTION
# ============================================================
with tab_predict:

    st.markdown("### 📋 Enter Patient Clinical Parameters")

    st.info(
        "This is an educational machine-learning prototype and "
        "does not provide a medical diagnosis."
    )

    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------
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

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --------------------------------------------------------
    # SAMPLE BUTTONS
    # --------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("🚨 High-Risk Sample", use_container_width=True):

            st.session_state.age = 63
            st.session_state.sex = 1
            st.session_state.cp = 0
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

    with c2:
        if st.button("💚 Healthy Sample", use_container_width=True):

            st.session_state.age = 42
            st.session_state.sex = 0
            st.session_state.cp = 1
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

    with c3:
        if st.button("🔄 Reset", use_container_width=True):

            for key, value in defaults.items():
                st.session_state[key] = value

            st.rerun()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    # ========================================================
    # COLUMN 1
    # ========================================================
    with col1:

        st.markdown("#### 👤 Demographics")

        age = st.slider(
            "Age",
            20,
            85,
            int(st.session_state.age)
        )

        sex_val = st.radio(
            "Sex",
            [1, 0],
            format_func=lambda x:
                "Male (1)" if x == 1 else "Female (0)",
            index=0 if st.session_state.sex == 1 else 1
        )

        cp_options = {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "Asymptomatic"
        }

        cp_val = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: cp_options[x],
            index=int(st.session_state.cp)
        )

        st.markdown("#### 🫀 Vessel / Thal")

        ca_val = st.selectbox(
            "Major Vessels (ca)",
            [0, 1, 2, 3],
            index=int(st.session_state.ca)
        )

        thal_options = {
            1: "Normal",
            2: "Fixed Defect",
            3: "Reversible Defect"
        }

        thal_val = st.selectbox(
            "Thalassemia (thal)",
            [1, 2, 3],
            format_func=lambda x: thal_options[x],
            index=[1, 2, 3].index(
                int(st.session_state.thal)
            )
        )

    # ========================================================
    # COLUMN 2
    # ========================================================
    with col2:

        st.markdown("#### 🩸 Vitals & Blood Chemistry")

        trestbps = st.slider(
            "Resting Blood Pressure",
            80,
            210,
            int(st.session_state.trestbps)
        )

        chol = st.slider(
            "Serum Cholesterol",
            120,
            580,
            int(st.session_state.chol),
            step=1
        )

        fbs_val = st.radio(
            "Fasting Blood Sugar > 120",
            [0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)",
            index=0 if st.session_state.fbs == 0 else 1
        )

        restecg_options = {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "LV Hypertrophy"
        }

        restecg_val = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x:
                restecg_options[x],
            index=int(st.session_state.restecg)
        )

    # ========================================================
    # COLUMN 3
    # ========================================================
    with col3:

        st.markdown("#### 🏃 Exercise Test")

        thalach = st.slider(
            "Maximum Heart Rate",
            70,
            210,
            int(st.session_state.thalach)
        )

        exang_val = st.radio(
            "Exercise-Induced Angina",
            [0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)",
            index=0 if st.session_state.exang == 0 else 1
        )

        oldpeak = st.slider(
            "ST Depression (oldpeak)",
            0.0,
            6.5,
            float(st.session_state.oldpeak),
            step=0.1
        )

        slope_options = {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }

        slope_val = st.selectbox(
            "ST Segment Slope",
            [0, 1, 2],
            format_func=lambda x:
                slope_options[x],
            index=int(st.session_state.slope)
        )

    # ========================================================
    # MODEL SELECTION
    # ========================================================
    st.write("---")

    available_models = list(all_models.keys())

    if not available_models and best_model_name:
        available_models = [best_model_name]

    if best_model_name in available_models:
        default_index = available_models.index(best_model_name)
    else:
        default_index = 0

    selected_model_name = st.selectbox(
        "🤖 Select Machine Learning Model",
        available_models,
        index=default_index
    )

    predict_clicked = st.button(
        "🔮 Analyze Heart Disease Risk",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # PREDICTION
    # ========================================================
    if predict_clicked:

        if selected_model_name == best_model_name:
            active_model = best_model
        else:
            active_model = all_models[selected_model_name]

        raw_features = pd.DataFrame(
            [[
                age,
                sex_val,
                cp_val,
                trestbps,
                chol,
                fbs_val,
                restecg_val,
                thalach,
                exang_val,
                oldpeak,
                slope_val,
                ca_val,
                thal_val
            ]],
            columns=feature_cols
        )

        if is_scaled and scaler is not None:
            model_input = scaler.transform(raw_features)
        else:
            model_input = raw_features

        prediction = int(active_model.predict(model_input)[0])

        if hasattr(active_model, "predict_proba"):
            probabilities = active_model.predict_proba(model_input)[0]
            risk_prob = float(probabilities[1])
        else:
            risk_prob = float(prediction)

        is_disease_likely = prediction == 1

        # ====================================================
        # RESULT
        # ====================================================
        st.write("---")
        st.markdown("### 📊 Prediction Result")

        if is_disease_likely:

            st.markdown(
                f"""
                <div class="result-card-high">
                <div class="risk-title-high">
                ⚠️ Higher Predicted Risk
                </div>

                <p>
                The selected machine-learning model classified
                this input as <b>heart disease = 1</b>.
                </p>

                <div class="risk-score">
                {risk_prob * 100:.1f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-card-low">
                <div class="risk-title-low">
                ✅ Lower Predicted Risk
                </div>

                <p>
                The selected machine-learning model classified
                this input as <b>heart disease = 0</b>.
                </p>

                <div class="risk-score">
                {risk_prob * 100:.1f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(risk_prob)

        st.caption(
            f"Model: **{selected_model_name}** | "
            f"Predicted probability: **{risk_prob * 100:.1f}%**"
        )

        st.warning(
            "This prediction is for academic demonstration only "
            "and must not be treated as a medical diagnosis."
        )

# ============================================================
# TAB 2 - MODEL BENCHMARK
# ============================================================
with tab_models:

    st.markdown("### 📊 Model Benchmark & Comparison")

    if results:

        results_df = pd.DataFrame(results)

        display_columns = [
            col for col in [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "cv_accuracy"
            ]
            if col in results_df.columns
        ]

        display_df = results_df[display_columns].copy()

        rename_map = {
            "model": "Algorithm",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-Score",
            "roc_auc": "ROC-AUC",
            "cv_accuracy": "CV Accuracy"
        }

        display_df.rename(
            columns=rename_map,
            inplace=True
        )

        for column in display_df.columns:

            if column != "Algorithm":

                display_df[column] = (
                    display_df[column] * 100
                ).round(2).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"🏆 Champion Model: **{best_model_name}** "
            f"with **{best_accuracy * 100:.2f}% accuracy**"
        )

        # Accuracy chart
        if "accuracy" in results_df.columns:

            chart_df = results_df[
                ["model", "accuracy"]
            ].copy()

            chart_df["accuracy"] *= 100

            chart_df.set_index(
                "model",
                inplace=True
            )

            st.markdown("#### 📈 Model Accuracy")

            st.bar_chart(chart_df)

    else:

        st.info("Model evaluation results are not available.")

# ============================================================
# TAB 3 - DATASET
# ============================================================
with tab_eda:

    st.markdown("### 📈 UCI Cleveland Dataset Explorer")

    if df_dataset is not None:

        total_records = len(df_dataset)

        disease_cases = int(
            df_dataset["target"].sum()
        )

        healthy_cases = (
            total_records - disease_cases
        )

        disease_pct = (
            disease_cases / total_records * 100
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Records",
            total_records
        )

        c2.metric(
            "Disease",
            f"{disease_cases} ({disease_pct:.1f}%)"
        )

        c3.metric(
            "Healthy",
            f"{healthy_cases}"
        )

        c4.metric(
            "Features",
            len(feature_cols)
        )

        st.write("---")

        st.dataframe(
            df_dataset.head(25),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### 📊 Target Distribution")

        target_counts = (
            df_dataset["target"]
            .value_counts()
            .sort_index()
        )

        target_counts.index = [
            "Healthy (0)"
            if x == 0
            else "Disease (1)"
            for x in target_counts.index
        ]

        st.bar_chart(target_counts)

    else:

        st.warning(
            "Dataset file was not found."
        )

# ============================================================
# TAB 4 - FEATURE GLOSSARY
# ============================================================
with tab_glossary:

    st.markdown("### 📚 Clinical Feature Glossary")

    glossary = {
        "age": "Age in years.",
        "sex": "Biological sex encoded as 0 or 1.",
        "cp": "Chest pain type.",
        "trestbps": "Resting blood pressure.",
        "chol": "Serum cholesterol.",
        "fbs": "Fasting blood sugar indicator.",
        "restecg": "Resting ECG result.",
        "thalach": "Maximum heart rate achieved.",
        "exang": "Exercise-induced angina indicator.",
        "oldpeak": "ST depression induced by exercise.",
        "slope": "Slope of peak exercise ST segment.",
        "ca": "Number of major vessels.",
        "thal": "Thallium stress-test result."
    }

    for feature, description in glossary.items():

        with st.expander(feature):

            st.write(description)

# ============================================================
# TAB 5 - DISCLAIMER
# ============================================================
with tab_disclaimer:

    st.markdown("### ⚠️ Medical & Educational Disclaimer")

    st.warning(
        """
        **EDUCATIONAL MACHINE LEARNING PROJECT ONLY**

        This application is an academic demonstration of
        machine-learning techniques for heart disease prediction.

        It is NOT a medical device and does NOT provide a medical
        diagnosis.

        Predictions should not be used as a substitute for advice
        from a qualified healthcare professional.

        The project is intended for educational and demonstration
        purposes only.
        """
    )

# ============================================================
# FOOTER
# ============================================================
st.write("---")

st.markdown(
    """
    <div style="text-align:center;color:#94a3b8;
    font-size:0.85rem;padding:10px 0;">
    CardioGuard AI — Heart Disease Prediction System |
    Built with Python, Streamlit, NumPy & Pandas |
    Educational Prototype
    </div>
    """,
    unsafe_allow_html=True
)
