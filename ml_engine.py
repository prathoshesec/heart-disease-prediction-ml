"""
Machine Learning Engine for Heart Disease Prediction
Fully vectorized, high-performance implementations using NumPy & Pandas.
Includes:
 - Preprocessing: StandardScaler, Stratified Train/Test Split
 - Algorithms:
     1. Logistic Regression (L2 regularized Gradient Descent)
     2. Random Forest Classifier (Bagged Decision Trees)
     3. Decision Tree Classifier (Gini Impurity)
     4. K-Nearest Neighbors (KNN)
     5. Gaussian Naive Bayes
 - Evaluation Metrics:
     - Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
     - 5-Fold Cross Validation
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (0=Female, 1=Male)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG (0-2)",
    "thalach": "Max Heart Rate Achieved (bpm)",
    "exang": "Exercise Induced Angina (0/1)",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "Slope of Peak ST Segment (0-2)",
    "ca": "Major Vessels Colored (0-3)",
    "thal": "Thalassemia (1-3)"
}

TARGET_COL = "target"

# ==========================================
# Preprocessing Utilities
# ==========================================

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    np.random.seed(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    if stratify:
        train_indices = []
        test_indices = []
        classes = np.unique(y)
        for c in classes:
            idx = np.where(y == c)[0]
            np.random.shuffle(idx)
            n_test = int(np.round(len(idx) * test_size))
            test_indices.extend(idx[:n_test])
            train_indices.extend(idx[n_test:])
        train_idx = np.array(train_indices)
        test_idx = np.array(test_indices)
    else:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        n_test = int(np.round(len(y) * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ==========================================
# Evaluation Metrics
# ==========================================

def accuracy_score(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def precision_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return float(2 * (p * r) / (p + r)) if (p + r) > 0 else 0.0

def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]

def roc_curve_and_auc(y_true, y_probs):
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    thresholds = np.linspace(1.0, 0.0, 101)
    tpr_list = []
    fpr_list = []
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5, [0.0, 1.0], [0.0, 1.0]

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        tpr_list.append(float(tp / n_pos))
        fpr_list.append(float(fp / n_neg))

    # Sort by FPR
    sorted_pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [float(p[0]) for p in sorted_pairs]
    tpr_sorted = [float(p[1]) for p in sorted_pairs]
    
    # Calculate trapezoidal AUC (NumPy 2.0+ compatible)
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    else:
        auc = sum(
            (fpr_sorted[i + 1] - fpr_sorted[i]) * (tpr_sorted[i + 1] + tpr_sorted[i]) / 2.0
            for i in range(len(fpr_sorted) - 1)
        )
    auc = max(0.0, min(1.0, abs(auc)))
    return auc, fpr_sorted, tpr_sorted


# ==========================================
# Machine Learning Classifiers
# ==========================================

class LogisticRegressionModel:
    def __init__(self, lr=0.05, n_iters=1000, l2_reg=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.l2_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias
        p1 = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances(self):
        abs_w = np.abs(self.weights)
        s = np.sum(abs_w)
        return abs_w / s if s > 0 else np.ones_like(abs_w) / len(abs_w)


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, prob=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.prob = prob

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeModel:
    def __init__(self, max_depth=5, min_samples_split=4, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self._grow_tree(X, y, depth=0)
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split):
            leaf_prob = np.mean(y == 1) if n_samples > 0 else 0.5
            leaf_val = 1 if leaf_prob >= 0.5 else 0
            return TreeNode(value=leaf_val, prob=leaf_prob)

        feat_idxs = np.arange(n_feats)
        if self.max_features is not None and self.max_features < n_feats:
            feat_idxs = np.random.choice(n_feats, self.max_features, replace=False)

        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
        if best_gain <= 1e-7 or best_feat is None:
            leaf_prob = np.mean(y == 1)
            return TreeNode(value=1 if leaf_prob >= 0.5 else 0, prob=leaf_prob)

        self.feature_importances_[best_feat] += best_gain * (n_samples / len(y))

        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return TreeNode(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_gini = self._gini(y)

        for feat in feat_idxs:
            X_column = X[:, feat]
            thresholds = np.percentile(X_column, np.linspace(10, 90, 9))
            for thresh in np.unique(thresholds):
                gain = self._information_gain(y, X_column, thresh, parent_gini)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh, best_gain

    def _information_gain(self, y, X_column, thresh, parent_gini):
        left_idxs = X_column <= thresh
        right_idxs = ~left_idxs
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = np.sum(left_idxs), np.sum(right_idxs)
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * g_l + (n_r / n) * g_r
        return parent_gini - child_gini

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        p1 = np.sum(y == 1) / m
        p0 = 1.0 - p1
        return 1.0 - (p0 ** 2 + p1 ** 2)

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.prob
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        probs = np.array([self._traverse_tree(x, self.root) for x in X])
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RandomForestModel:
    def __init__(self, n_trees=50, max_depth=6, min_samples_split=4, max_features="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_feats = X.shape
        self.trees = []

        if self.max_features == "sqrt":
            max_f = max(1, int(np.sqrt(n_feats)))
        else:
            max_f = n_feats

        total_importances = np.zeros(n_feats)

        for _ in range(self.n_trees):
            # Bootstrap sample
            boot_idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[boot_idxs], y[boot_idxs]
            tree = DecisionTreeModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_f
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            total_importances += tree.feature_importances_

        s = np.sum(total_importances)
        self.feature_importances_ = total_importances / s if s > 0 else np.ones(n_feats) / n_feats
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        all_tree_probs = np.array([tree.predict_proba(X)[:, 1] for tree in self.trees])
        avg_prob = np.mean(all_tree_probs, axis=0)
        return np.column_stack([1.0 - avg_prob, avg_prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances(self):
        return self.feature_importances_


class KNNModel:
    def __init__(self, k=7):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.int64)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        probs = []
        for x in X:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_indices]
            probs.append(np.mean(k_labels == 1))
        p1 = np.array(probs)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GaussianNBModel:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-4
            self.priors[c] = len(X_c) / len(y)
        return self

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        probs = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                conditional = np.sum(np.log(self._pdf(c, x) + 1e-9))
                posteriors.append(prior + conditional)
            # Softmax to get calibrated probabilities
            posteriors = np.array(posteriors)
            exp_post = np.exp(posteriors - np.max(posteriors))
            probs.append(exp_post / np.sum(exp_post))
        return np.array(probs)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ==========================================
# Orchestration & Training
# ==========================================

def create_model_instance(name):
    if name == "Random Forest Classifier":
        return RandomForestModel(n_trees=60, max_depth=6)
    elif name == "Logistic Regression":
        return LogisticRegressionModel(lr=0.08, n_iters=1200, l2_reg=0.02)
    elif name == "Decision Tree":
        return DecisionTreeModel(max_depth=5)
    elif name == "K-Nearest Neighbors (KNN)":
        return KNNModel(k=7)
    elif name == "Gaussian Naive Bayes":
        return GaussianNBModel()
    raise ValueError(f"Unknown model name {name}")


def run_training_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "heart.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        from download_or_create_data import generate_cleveland_dataset
        df = generate_cleveland_dataset()
    else:
        df = pd.read_csv(data_path)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / Test Split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate Models
    models = {
        "Random Forest Classifier": {
            "model": RandomForestModel(n_trees=60, max_depth=6),
            "use_scaled": False
        },
        "Logistic Regression": {
            "model": LogisticRegressionModel(lr=0.08, n_iters=1200, l2_reg=0.02),
            "use_scaled": True
        },
        "Decision Tree": {
            "model": DecisionTreeModel(max_depth=5),
            "use_scaled": False
        },
        "K-Nearest Neighbors (KNN)": {
            "model": KNNModel(k=7),
            "use_scaled": True
        },
        "Gaussian Naive Bayes": {
            "model": GaussianNBModel(),
            "use_scaled": True
        }
    }

    results = []
    confusion_matrices = {}
    roc_curves = {}

    print("=" * 80)
    print("HEART DISEASE PREDICTION - MULTI-MODEL BENCHMARK")
    print("=" * 80)
    print(f"{'Algorithm':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 80)

    best_model_name = None
    best_score = -1.0
    trained_models = {}

    for name, config in models.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc, fpr, tpr = roc_curve_and_auc(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # 5-Fold Cross Validation
        fold_accs = []
        n_samples = len(X)
        fold_size = n_samples // 5
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        for fold in range(5):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            tr_idx = np.setdiff1d(indices, val_idx)
            fold_clf = create_model_instance(name)
            X_cv_tr = X_scaled[tr_idx] if use_scaled else X[tr_idx]
            X_cv_val = X_scaled[val_idx] if use_scaled else X[val_idx]
            fold_clf.fit(X_cv_tr, y[tr_idx])
            fold_accs.append(accuracy_score(y[val_idx], fold_clf.predict(X_cv_val)))

        cv_mean = float(np.mean(fold_accs))

        trained_models[name] = clf
        confusion_matrices[name] = cm
        roc_curves[name] = {"fpr": fpr, "tpr": tpr, "auc": round(auc, 4)}

        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_accuracy": round(cv_mean, 4),
            "use_scaled": use_scaled
        })

        print(f"{name:<28} | {acc * 100:6.2f}%   | {prec * 100:6.2f}%   | {rec * 100:6.2f}%   | {f1:6.4f}   | {auc:6.4f}")

        # Choose best based on balanced F1 and accuracy
        score = (f1 + acc) / 2.0
        if score > best_score:
            best_score = score
            best_model_name = name

    print("-" * 80)
    print(f"SELECTED CHAMPION MODEL: {best_model_name} (Combined Score: {best_score:.4f})")

    # Feature Importance extraction
    champion = trained_models[best_model_name]
    feature_importance = {}
    if hasattr(champion, "feature_importances"):
        imps = champion.feature_importances
    elif hasattr(champion, "feature_importances_"):
        imps = champion.feature_importances_
    else:
        # Fallback to Random Forest feature importance
        rf = trained_models["Random Forest Classifier"]
        imps = rf.feature_importances_

    for col, imp in zip(FEATURE_COLS, imps):
        feature_importance[col] = round(float(imp), 4)

    # Sort descending
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    # Save Bundle
    bundle = {
        "best_model_name": best_model_name,
        "best_model": champion,
        "use_scaled": models[best_model_name]["use_scaled"],
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "results": results,
        "confusion_matrices": confusion_matrices,
        "roc_curves": roc_curves,
        "feature_importance": feature_importance,
        "trained_models": trained_models
    }

    bundle_file = os.path.join(models_dir, "heart_disease_bundle.pkl")
    with open(bundle_file, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved complete model bundle to: {bundle_file}")

    # Metrics JSON summary
    summary_file = os.path.join(models_dir, "metrics_summary.json")
    summary = {
        "best_model_name": best_model_name,
        "results": results,
        "confusion_matrices": confusion_matrices,
        "feature_importance": feature_importance,
        "dataset_size": len(df),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "target_distribution": {
            "low_risk_0": int(np.sum(y == 0)),
            "high_risk_1": int(np.sum(y == 1))
        }
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary JSON to: {summary_file}")

    return bundle

if __name__ == "__main__":
    run_training_pipeline()
