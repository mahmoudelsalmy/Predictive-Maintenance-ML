import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import joblib
import json
from datetime import datetime

# =========================
# LOAD DATA
# =========================
healthy = pd.read_csv("Data/fullHealthy.csv")
broken  = pd.read_csv("Data/fullBroken.csv")


SENSORS = ["a1", "a2", "a3", "a4"]

# =========================
# SLIDING WINDOW
# =========================
def create_windows(signal, window_size=256, step=128):
    return [
        signal[i:i+window_size]
        for i in range(0, len(signal) - window_size, step)
    ]

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(window):
    return [
        np.mean(window),
        np.std(window),
        np.sqrt(np.mean(window**2)),   # RMS
        np.min(window),
        np.max(window),
        skew(window),
        kurtosis(window)
    ]

def process_dataframe(df):
    sensor_windows = []
    for col in SENSORS:
        sensor_windows.append(create_windows(df[col].values))

    X = []
    num_windows = min(len(w) for w in sensor_windows)

    for i in range(num_windows):
        feats = []
        for sw in sensor_windows:
            feats.extend(extract_features(sw[i]))
        X.append(feats)

    return np.array(X)

# =========================
# BUILD DATASET
# =========================
print("🔄 Extracting features...")

X_healthy = process_dataframe(healthy)
X_broken  = process_dataframe(broken)

y_healthy = np.zeros(len(X_healthy))
y_broken  = np.ones(len(X_broken))

X = np.vstack([X_healthy, X_broken])
y = np.hstack([y_healthy, y_broken])

print("✅ Feature matrix shape:", X.shape)

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =========================
# PCA
# =========================
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)

print("🔽 After PCA:")
print("Train:", X_train_pca.shape)
print("Test :", X_test_pca.shape)

# =========================
# MODELS
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=50,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),
    "SVM (RBF)": SVC(
        C=10,
        gamma="scale",
        probability=True   
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    )
}

# =========================
# TRAIN + TEST
# =========================
results = {}

print("\n📊 Model Evaluation:\n")

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("=" * 60)
    print(name)
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred))

# =========================
# SELECT BEST MODEL
# =========================

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\n🏆 Best Model:", best_model_name)
print(f"🎯 Test Accuracy: {best_accuracy:.2%}")

# =========================
# RETRAIN BEST MODEL ON FULL DATA
# =========================
print("\n🔁 Retraining best model on FULL dataset...")

best_model = models[best_model_name]

X_all_s = scaler.fit_transform(X)
X_all_pca = pca.fit_transform(X_all_s)

best_model.fit(X_all_pca, y)

print("✅ Final model trained on full dataset")

# =========================
# SAVE FILES
# =========================
joblib.dump(best_model, "trained_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

FEATURE_NAMES = ["a1", "a2", "a3", "a4"]


metadata = {
    "model_type": best_model_name,
    "accuracy": best_accuracy,          
    "test_accuracy": best_accuracy,     
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "features_per_sensor": 7,
    "sensors": SENSORS,
    "feature_names": FEATURE_NAMES,
    "target_classes": ["healthy", "broken"],
    "trained_date": datetime.now().isoformat()
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n💾 Model, scaler, PCA, and metadata saved successfully")

