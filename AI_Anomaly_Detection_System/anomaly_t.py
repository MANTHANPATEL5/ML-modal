
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET = "creditcard.csv"

MODEL_FILE = "anomaly_model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_FILE = "feature_info.pkl"

RANDOM_STATE = 42

CONTAMINATION = 0.0017


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 70)
print("AI ANOMALY DETECTION SYSTEM")
print("MODEL TRAINING")
print("=" * 70)


if not os.path.exists(DATASET):

    raise FileNotFoundError(
        "creditcard.csv not found. "
        "Put it in the same folder as train_model.py."
    )


df = pd.read_csv(
    DATASET
)


print("\nDataset loaded successfully!")

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ==========================================================
# DATA VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

df = df.drop_duplicates(
    ignore_index=True
)


# ==========================================================
# REMOVE TARGET
# ==========================================================

if "Class" in df.columns:

    y_true = df["Class"].copy()

    X = df.drop(
        columns=["Class"]
    )

else:

    y_true = None

    X = df.copy()


# ==========================================================
# AUTOMATIC FEATURE SELECTION
# ==========================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()


if len(numeric_features) == 0:

    raise ValueError(
        "No numerical features found."
    )


X = X[
    numeric_features
].copy()


print("\nSelected features:")

for feature in numeric_features:

    print(
        "✓",
        feature
    )


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
# SCALING
# ==========================================================

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


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

model.fit(
    X_scaled
)


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

feature_info = {
    "features": numeric_features,
    "contamination": CONTAMINATION,
    "random_state": RANDOM_STATE
}

joblib.dump(
    feature_info,
    FEATURE_FILE
)


# ==========================================================
# TRAINING PREDICTION
# ==========================================================

predictions = model.predict(
    X_scaled
)


anomaly_labels = np.where(
    predictions == -1,
    1,
    0
)


total = len(
    anomaly_labels
)

anomalies = int(
    anomaly_labels.sum()
)

normal = (
    total - anomalies
)


# ==========================================================
# RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    "Total records:",
    total
)

print(
    "Normal:",
    normal
)

print(
    "Anomalies:",
    anomalies
)

print(
    "Anomaly rate:",
    round(
        anomalies / total * 100,
        3
    ),
    "%"
)


# ==========================================================
# OPTIONAL EVALUATION
# ==========================================================

if y_true is not None:

    # Align target after duplicate removal
    y_eval = y_true.loc[
        X.index
    ]

    actual_fraud = int(
        y_eval.sum()
    )

    detected_fraud = int(
        (
            (y_eval == 1)
            &
            (anomaly_labels == 1)
        ).sum()
    )

    print(
        "\nKnown fraud records:",
        actual_fraud
    )

    print(
        "Detected known fraud:",
        detected_fraud
    )

    if actual_fraud > 0:

        recall = (
            detected_fraud
            / actual_fraud
        )

        print(
            "Fraud recall:",
            round(
                recall * 100,
                2
            ),
            "%"
        )


# ==========================================================
# FILES
# ==========================================================

print("\nCreated files:")

print(
    "✓",
    MODEL_FILE
)

print(
    "✓",
    SCALER_FILE
)

print(
    "✓",
    FEATURE_FILE
)

print("\nModel ready.")
