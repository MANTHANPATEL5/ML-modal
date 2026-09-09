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

MAX_TRAIN_ROWS = 50000
CHART_SAMPLE_SIZE = 5000
DISPLAY_ROWS = 100

N_ESTIMATORS = 100
RANDOM_STATE = 42


# ==========================================================
# CSS
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

st.sidebar.success("AI Model Ready")

st.sidebar.write("**Algorithm:** Isolation Forest")
st.sidebar.write("**Detection:** Unsupervised")
st.sidebar.write(f"**Trees:** {N_ESTIMATORS}")
st.sidebar.write(
    f"**Training Limit:** {MAX_TRAIN_ROWS:,} rows"
)

st.sidebar.markdown("---")

st.sidebar.info(
    "The system automatically selects numeric features "
    "and excludes common target/label columns."
)


# ==========================================================
# LOAD CSV
# ==========================================================

@st.cache_data(show_spinner=False)
def load_csv(file_bytes):

    return pd.read_csv(
        BytesIO(file_bytes)
    )


# ==========================================================
# TARGET COLUMN DETECTION
# ==========================================================

def detect_target_columns(df):

    target_names = {
        "class",
        "target",
        "label",
        "fraud",
        "is_fraud",
        "anomaly",
        "prediction",
        "output"
    }

    excluded = []

    for column in df.columns:

        name = str(column).strip().lower()

        if name in target_names:
            excluded.append(column)

    return excluded


# ==========================================================
# AUTOMATIC FEATURE SELECTION
# ==========================================================

def select_features(df):

    target_columns = detect_target_columns(df)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    features = [
        column
        for column in numeric_columns
        if column not in target_columns
    ]

    # Remove common ID columns
    id_columns = {
        "id",
        "index",
        "transaction_id",
        "customer_id",
        "user_id"
    }

    filtered_features = []

    for column in features:

        name = str(column).strip().lower()

        if name not in id_columns:
            filtered_features.append(column)

    if filtered_features:
        features = filtered_features

    return features, target_columns


# ==========================================================
# PREPARE FEATURES
# ==========================================================

def prepare_features(df, features):

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        X[column] = X[column].fillna(
            median_value
        )

    return X


# ==========================================================
# TRAIN ISOLATION FOREST
# ==========================================================

def train_model(X_train):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_scaled
    )

    return model, scaler


# ==========================================================
# FILE UPLOAD
# ==========================================================

st.header("📁 Upload CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help="Upload a CSV containing numeric business, transaction, or IoT data."
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
Feature Scaling
     ↓
Isolation Forest
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
