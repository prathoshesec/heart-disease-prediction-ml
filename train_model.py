import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"

def load_and_preprocess_data():
    if not os.path.exists(DATA_PATH):
        from download_or_create_data import fetch_or_generate_dataset
        df = fetch_or_generate_dataset()
    else:
        df = pd.read_csv(DATA_PATH)

    # Basic data cleaning
    # Ensure correct column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle any potential target alias (e.g., 'num' or 'output')
    if "target" not in df.columns:
        for alt in ["output", "condition", "num", "diagnosis"]:
            if alt in df.columns:
                df.rename(columns={alt: "target"}, inplace=True)
                break

    # If target has values > 1 (e.g. original Cleveland 0-4), binarize: > 0 means disease
    if df["target"].max() > 1:
        df["target"] = (df["target"] > 0).astype(int)

    # Ensure all required features are present
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Drop missing values if any
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df

def train_and_evaluate():
    print("=" * 60)
    print("HEART DISEASE PREDICTION - MODEL TRAINING & EVALUATION")
    print("=" * 60)

    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['target'].value_counts().to_dict()}\n")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidates
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, random_state=42),
        "Support Vector Machine": SVC(probability=True, kernel="rbf", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    results = []
    trained_instances = {}
    confusion_matrices = {}

    print(f"{'Model':<26} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1':<7} | {'ROC-AUC':<7}")
    print("-" * 78)

    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        # Train
        if name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            # Tree models work well directly on raw or scaled features
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        trained_instances[name] = model
        confusion_matrices[name] = cm

        results.append({
            "model": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "cv_mean_acc": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4)
        })

        print(f"{name:<26} | {acc * 100:6.2f}%   | {prec * 100:6.2f}%   | {rec * 100:6.2f}%   | {f1:6.4f} | {roc:6.4f}")

        # Choose best based on balanced F1 score and Accuracy
        combined_metric = (f1 + acc) / 2.0
        if combined_metric > best_f1:
            best_f1 = combined_metric
            best_model_name = name

    print("-" * 78)
    print(f"Top Performing Model: {best_model_name} (Combined F1+Acc Score: {best_f1:.4f})")

    best_model = trained_instances[best_model_name]
    is_scaled = best_model_name not in ["Random Forest", "Gradient Boosting", "Decision Tree"]

    # Calculate feature importance
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        for col, imp in zip(FEATURE_COLS, best_model.feature_importances_):
            feature_importance[col] = round(float(imp), 4)
    elif hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_[0])
        total = np.sum(coefs) or 1.0
        for col, imp in zip(FEATURE_COLS, coefs / total):
            feature_importance[col] = round(float(imp), 4)
    else:
        # Default fallback to Random Forest feature importance
        rf = trained_instances["Random Forest"]
        for col, imp in zip(FEATURE_COLS, rf.feature_importances_):
            feature_importance[col] = round(float(imp), 4)

    # Sort feature importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    # Save artifacts
    bundle = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "is_scaled": is_scaled,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "results": results,
        "confusion_matrices": confusion_matrices,
        "feature_importance": feature_importance,
        "all_models": trained_instances
    }

    model_file = os.path.join(MODELS_DIR, "heart_disease_bundle.joblib")
    joblib.dump(bundle, model_file)
    print(f"\nSaved model bundle to: {model_file}")

    # Also save json summary for quick web consumption
    metrics_summary_file = os.path.join(MODELS_DIR, "metrics_summary.json")
    summary = {
        "best_model_name": best_model_name,
        "results": results,
        "feature_importance": feature_importance,
        "confusion_matrices": confusion_matrices,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "feature_cols": FEATURE_COLS
    }
    with open(metrics_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to: {metrics_summary_file}")

    return bundle

if __name__ == "__main__":
    train_and_evaluate()
