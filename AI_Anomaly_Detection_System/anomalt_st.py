import os
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Anomaly Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DISPLAY_ROWS = 100

# Maximum rows used for ML training.
# This keeps large datasets such as creditcard.csv responsive.
MAX_TRAIN_ROWS = 50000

# Maximum rows used for charts.
CHART_SAMPLE_SIZE = 5000

# Isolation Forest configuration
N_ESTIMATORS = 100
CONTAMINATION = "auto"
RANDOM_STATE = 42


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777777;
        font-size: 18px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #dddddd;
        padding: 12px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">🚨 AI Anomaly Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect unusual patterns using Artificial Intelligence '
    'and Isolation Forest'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.info(
    """
    The system automatically:

    • Validates the CSV
    • Selects numeric features
    • Removes target columns
    • Handles missing values
    • Scales the data
    • Trains Isolation Forest
    • Detects anomalies
    • Generates explanations
    """
)

st.sidebar.markdown("---")

st.sidebar.write(
    "**Algorithm:** Isolation Forest"
)

st.sidebar.write(
    "**Detection:** Unsupervised"
)

st.sidebar.write(
    f"**Trees:** {N_ESTIMATORS}"
)

st.sidebar.write(
    f"**Training Limit:** {MAX_TRAIN_ROWS:,} rows"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "No pre-trained .pkl files are required."
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def detect_target_columns(df):
    """
    Detect columns that should not be used as anomaly features.
    """

    target_names = [
        "class",
        "target",
        "label",
        "fraud",
        "is_fraud",
        "anomaly",
        "prediction",
        "output"
    ]

    excluded = []

    for column in df.columns:

        clean_name = str(column).strip().lower()

        if clean_name in target_names:
            excluded.append(column)

    return excluded


# ==========================================================
# AUTOMATIC FEATURE SELECTION
# ==========================================================

def automatic_feature_selection(df):
    """
    Automatically select numeric columns while removing
    target/label columns.
    """

    target_columns = detect_target_columns(df)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    features = [
        column
        for column in numeric_columns
        if column not in target_columns
    ]

    # Remove obvious ID/index columns when possible.
    id_names = [
        "id",
        "index",
        "transaction_id",
        "customer_id",
        "user_id"
    ]

    filtered_features = []

    for column in features:

        clean_name = str(column).strip().lower()

        if clean_name in id_names:
            continue

        filtered_features.append(column)

    # If removing ID columns leaves nothing,
    # use all numeric columns except targets.
    if len(filtered_features) > 0:
        features = filtered_features

    return features, target_columns


# ==========================================================
# PREPARE FEATURES
# ==========================================================

def prepare_features(df, features):
    """
    Prepare numeric feature matrix.
    """

    X = df[features].copy()

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Convert everything to numeric
    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Fill missing values using median
    for column in features:

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        X[column] = X[column].fillna(
            median_value
        )

    return X


# ==========================================================
# MODEL TRAINING
# ==========================================================

@st.cache_resource(show_spinner=False)
def train_ai_model(
    training_data,
    feature_names
):
    """
    Train Isolation Forest automatically.

    The model is trained only once for the uploaded dataset
    because Streamlit caches the model.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        training_data
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train_scaled
    )

    return model, scaler


# ==========================================================
# PROCESS DATASET
# ==========================================================

@st.cache_data(show_spinner=False)
def load_csv(file_bytes):

    df = pd.read_csv(
        BytesIO(file_bytes)
    )

    return df


# ==========================================================
# CSV UPLOAD
# ==========================================================

st.header("📁 Upload CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help=(
        "Upload creditcard.csv or any other "
        "numeric business/IoT dataset."
    )
)


# ==========================================================
# NO FILE
# ==========================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a CSV file to start AI anomaly detection."
    )

    st.markdown(
        """
### 🔄 AI Machine Learning Pipeline

```text
CSV Dataset
     ↓
Data Validation
     ↓
Automatic Feature Selection
     ↓
Target Column Removal
     ↓
Missing Value Handling
     ↓
Standard Scaling
     ↓
Isolation Forest Training
     ↓
Anomaly Score
     ↓
Normal / Anomaly
     ↓
AI Explanation
     ↓
Visualization
     ↓
Download Results
