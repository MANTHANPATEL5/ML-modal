import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================================
# SETTINGS
# ==========================================================

DATASET_FILE = "traffic_data.csv"
MODEL_FILE = "traffic_congestion_model.pkl"

TARGET_COLUMN = "label"


# ==========================================================
# TOP 10 IMPORTANT FEATURES
# ==========================================================

SELECTED_FEATURES = [
    "speed_density_ratio",
    "channel_busy_ratio_pct",
    "flow_veh_per_hr",
    "packet_loss_pct",
    "visibility_km",
    "avg_speed_kmph",
    "heading_deg",
    "incident_num",
    "wireless_congestion_intensity",
    "avg_wait_time_s"
]


# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("🚦 TRAFFIC CONGESTION PREDICTION")
print("=" * 70)


# ==========================================================
# LOAD DATASET
# ==========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ==========================================================
# CHECK TARGET
# ==========================================================

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found."
    )


# ==========================================================
# CHECK FEATURES
# ==========================================================

missing_features = [
    feature
    for feature in SELECTED_FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "These selected features are missing:\n"
        + "\n".join(missing_features)
    )


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df[SELECTED_FEATURES].copy()

y = df[TARGET_COLUMN].copy()


# ==========================================================
# CONVERT TO NUMERIC
# ==========================================================

for feature in SELECTED_FEATURES:

    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

missing_values = X.isnull().sum().sum()

print("\nMissing values:", missing_values)

if missing_values > 0:

    X = X.fillna(
        X.median()
    )


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================================
# RANDOM FOREST
# ==========================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(

    n_estimators=300,

    max_features="sqrt",

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


# ==========================================================
# TRAIN
# ==========================================================

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================================
# TEST
# ==========================================================

y_pred = model.predict(
    X_test
)


# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print(cm)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_importance = pd.DataFrame({

    "Feature":
        SELECTED_FEATURES,

    "Importance":
        model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("FINAL TOP 10 FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance.to_string(
        index=False
    )
)


# ==========================================================
# SAVE MODEL
# ==========================================================

MODEL_PACKAGE = {

    "model": model,

    "features": SELECTED_FEATURES,

    "classes": model.classes_.tolist(),

    "accuracy": accuracy,

    "feature_importance":
        feature_importance.to_dict(
            orient="records"
        )
}


joblib.dump(
    MODEL_PACKAGE,
    MODEL_FILE
)


# ==========================================================
# FINAL
# ==========================================================

print("\n" + "=" * 70)
print("✅ MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print("\nModel file:", MODEL_FILE)

print("\nSelected features: 10")

for feature in SELECTED_FEATURES:

    print("✓", feature)

print("\nRemoved features: 14")

all_features = [
    column
    for column in df.columns
    if column not in SELECTED_FEATURES
    and column != TARGET_COLUMN
    and column not in ["timestamp", "road_segment_id"]
]

for feature in all_features:

    print("✗", feature)

print("\n🚦 TOP 10 FEATURE MODEL READY!")