import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET = "creditcard.csv"

MODEL_FILE = "anomaly_model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_FILE = "feature_info.pkl"

CONTAMINATION = 0.0017
RANDOM_STATE = 42


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("AI ANOMALY DETECTION - MODEL TRAINING")
print("=" * 60)

if not os.path.exists(DATASET):
    raise FileNotFoundError(
        f"{DATASET} not found. Put creditcard.csv in this folder."
    )

df = pd.read_csv(DATASET)

print("\nDataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ==========================================================
# DATA VALIDATION
# ==========================================================

print("\nChecking dataset...")

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)


# ==========================================================
# REMOVE TARGET COLUMN
# ==========================================================

# Class is the known fraud label.
# It is NOT used for training the unsupervised model.

if "Class" in df.columns:
    y_true = df["Class"].copy()
    X = df.drop(columns=["Class"])
else:
    y_true = None
    X = df.copy()


# ==========================================================
# AUTOMATIC FEATURE SELECTION
# ==========================================================

# Keep numeric columns automatically
numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

if len(numeric_features) == 0:
    raise ValueError("No numerical features found in dataset.")

X = X[numeric_features]

print("\nSelected features:")
for feature in numeric_features:
    print(" -", feature)


# ==========================================================
# MISSING VALUE HANDLING
# ==========================================================

for column in numeric_features:
    X[column] = X[column].replace(
        [np.inf, -np.inf],
        np.nan
    )

    X[column] = X[column].fillna(
        X[column].median()
    )


# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================================
# ISOLATION FOREST
# ==========================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_scaled)


# ==========================================================
# TRAINING PREDICTIONS
# ==========================================================

predictions = model.predict(X_scaled)

# Isolation Forest:
# -1 = anomaly
#  1 = normal

anomaly_labels = np.where(
    predictions == -1,
    1,
    0
)

# Larger value = more normal
decision_scores = model.decision_function(X_scaled)

# Convert into easier anomaly score
anomaly_scores = -decision_scores


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

feature_info = {
    "features": numeric_features,
    "contamination": CONTAMINATION,
    "random_state": RANDOM_STATE
}

joblib.dump(feature_info, FEATURE_FILE)


# ==========================================================
# RESULTS
# ==========================================================

total = len(anomaly_labels)
anomalies = int(anomaly_labels.sum())
normal = total - anomalies

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("Total records :", total)
print("Normal records:", normal)
print("Anomalies     :", anomalies)
print(
    "Anomaly rate  :",
    round((anomalies / total) * 100, 3),
    "%"
)

if y_true is not None:

    actual_fraud = int(y_true.sum())

    detected_fraud = int(
        ((y_true == 1) & (anomaly_labels == 1)).sum()
    )

    print("\nKnown fraud records:", actual_fraud)
    print("Detected known fraud:", detected_fraud)

    if actual_fraud > 0:
        recall = detected_fraud / actual_fraud

        print(
            "Fraud detection recall:",
            round(recall * 100, 2),
            "%"
        )


print("\nSaved files:")
print(MODEL_FILE)
print(SCALER_FILE)
print(FEATURE_FILE)

print("\nModel is ready.")