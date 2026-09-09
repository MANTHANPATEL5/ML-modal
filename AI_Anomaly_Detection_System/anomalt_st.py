import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st


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

# IMPORTANT:
# Always use the directory where this Python file exists.
# This fixes Streamlit Cloud path problems.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "anomaly_model.pkl"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "feature_info.pkl"
)


# Maximum rows displayed in tables
DISPLAY_ROWS = 100

# Maximum rows used for charts
CHART_SAMPLE_SIZE = 5000


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
    'Detect unusual transactions using Artificial Intelligence '
    'and Isolation Forest'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# MODEL FILE CHECK
# ==========================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_FILE
]

missing_files = [
    file
    for file in required_files
    if not os.path.isfile(file)
]


if missing_files:

    st.error(
        "❌ Trained model files are missing."
    )

    st.write(
        "The application expects these files in the same "
        "folder as anomalt_st.py:"
    )

    for file in missing_files:

        st.write(
            f"• `{os.path.basename(file)}`"
        )

    st.info(
        "Make sure anomaly_model.pkl, scaler.pkl and "
        "feature_info.pkl are uploaded to the "
        "AI_Anomaly_Detection_System folder."
    )

    st.code(
        BASE_DIR,
        language="text"
    )

    st.stop()


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    feature_info = joblib.load(
        FEATURE_FILE
    )

    return model, scaler, feature_info


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model, scaler, feature_info = load_model_files()

    # ------------------------------------------------------
    # Read features from feature_info.pkl
    # ------------------------------------------------------

    if isinstance(feature_info, dict):

        if "features" in feature_info:

            features = feature_info["features"]

        elif "feature_names" in feature_info:

            features = feature_info["feature_names"]

        else:

            raise ValueError(
                "feature_info.pkl does not contain "
                "'features' or 'feature_names'."
            )

    elif isinstance(feature_info, (list, tuple)):

        features = list(feature_info)

    else:

        raise ValueError(
            "Unsupported feature_info.pkl format."
        )

    features = list(features)


except Exception as e:

    st.error(
        "❌ Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ==========================================================
# SIDEBAR MODEL INFORMATION
# ==========================================================

st.sidebar.title(
    "⚙️ Model Information"
)

st.sidebar.success(
    "Model Loaded"
)

st.sidebar.write(
    "**Algorithm:** Isolation Forest"
)

st.sidebar.write(
    "**Detection:** Unsupervised"
)

st.sidebar.write(
    f"**Features:** {len(features)}"
)

if hasattr(model, "n_estimators"):

    st.sidebar.write(
        f"**Trees:** {model.n_estimators}"
    )

st.sidebar.markdown("---")

st.sidebar.info(
    "The Class column is not used as a model input. "
    "The AI detects unusual transaction patterns "
    "without using fraud labels."
)

st.sidebar.markdown("---")

st.sidebar.write(
    "**Model Features**"
)

with st.sidebar.expander(
    "View features"
):

    for feature in features:

        st.write(
            f"• {feature}"
        )


# ==========================================================
# DATA PROCESSING
# ==========================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3
)
def process_dataset(
    file_bytes,
    file_name
):

    # ------------------------------------------------------
    # LOAD CSV
    # ------------------------------------------------------

    df = pd.read_csv(
        BytesIO(file_bytes)
    )


    # ------------------------------------------------------
    # BASIC VALIDATION
    # ------------------------------------------------------

    rows = len(df)

    columns = len(df.columns)

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )


    # ------------------------------------------------------
    # CHECK REQUIRED FEATURES
    # ------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        return {
            "error": "missing_features",
            "missing_features": missing_features
        }


    # ------------------------------------------------------
    # SELECT MODEL FEATURES
    # ------------------------------------------------------

    X = df[
        features
    ].copy()


    # ------------------------------------------------------
    # CONVERT FEATURES TO NUMERIC
    # ------------------------------------------------------

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


    # ------------------------------------------------------
    # REPLACE INFINITE VALUES
    # ------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # ------------------------------------------------------
    # MISSING VALUE HANDLING
    # ------------------------------------------------------

    for column in features:

        median_value = X[column].median()

        if pd.isna(median_value):

            median_value = 0.0

        X[column] = X[column].fillna(
            median_value
        )


    # ------------------------------------------------------
    # SCALING
    # ------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )


    # ------------------------------------------------------
    # ISOLATION FOREST PREDICTION
    # ------------------------------------------------------

    predictions = model.predict(
        X_scaled
    )


    # ------------------------------------------------------
    # ANOMALY SCORE
    # ------------------------------------------------------

    decision_scores = model.decision_function(
        X_scaled
    )

    # Lower Isolation Forest decision score
    # = more anomalous.

    anomaly_scores = -decision_scores


    # ------------------------------------------------------
    # CREATE RESULTS
    # ------------------------------------------------------

    results = df.copy()

    results["Anomaly Score"] = anomaly_scores

    results["Prediction"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )


    # ------------------------------------------------------
    # TOTAL COUNTS
    # ------------------------------------------------------

    total_records = len(results)

    anomaly_count = int(
        (
            results["Prediction"]
            == "Anomaly"
        ).sum()
    )

    normal_count = (
        total_records
        - anomaly_count
    )


    # ------------------------------------------------------
    # ANOMALY RATE
    # ------------------------------------------------------

    if total_records > 0:

        anomaly_rate = (
            anomaly_count
            / total_records
            * 100
        )

    else:

        anomaly_rate = 0.0


    # ------------------------------------------------------
    # GET ANOMALIES
    # ------------------------------------------------------

    anomalies = results[
        results["Prediction"]
        == "Anomaly"
    ].copy()


    # ------------------------------------------------------
    # SORT MOST SUSPICIOUS FIRST
    # ------------------------------------------------------

    anomalies = anomalies.sort_values(
        "Anomaly Score",
        ascending=False
    )


    # ------------------------------------------------------
    # RESET ANOMALY INDEX
    # ------------------------------------------------------

    anomalies = anomalies.reset_index(
        drop=False
    )

    anomalies.rename(
        columns={
            "index": "Original Row"
        },
        inplace=True
    )


    # ------------------------------------------------------
    # CHART SAMPLE
    # ------------------------------------------------------

    if len(results) > CHART_SAMPLE_SIZE:

        chart_results = results[
            ["Anomaly Score"]
        ].sample(
            n=CHART_SAMPLE_SIZE,
            random_state=42
        )

    else:

        chart_results = results[
            ["Anomaly Score"]
        ].copy()


    chart_results = chart_results.sort_values(
        "Anomaly Score"
    ).reset_index(
        drop=True
    )


    # ------------------------------------------------------
    # RETURN DATA
    # ------------------------------------------------------

    return {

        "error": None,

        "df": df,

        "X": X,

        "results": results,

        "anomalies": anomalies,

        "chart_results": chart_results,

        "rows": rows,

        "columns": columns,

        "missing_values": missing_values,

        "duplicates": duplicates,

        "total_records": total_records,

        "normal_count": normal_count,

        "anomaly_count": anomaly_count,

        "anomaly_rate": anomaly_rate
    }


# ==========================================================
# CSV UPLOAD
# ==========================================================

st.header(
    "📁 Upload CSV Dataset"
)

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help=(
        "Upload a CSV containing the features "
        "used during model training."
    )
)


# ==========================================================
# NO FILE
# ==========================================================

if uploaded_file is None:

    st.info(
        "👆 Upload your CSV file to start "
        "AI anomaly detection."
    )

    st.markdown(
        """
        ### 🔄 AI Machine Learning Pipeline

        ```text
        CSV Dataset
             ↓
        Data Validation
             ↓
        Missing-Value Handling
             ↓
        Numeric Conversion
             ↓
        Feature Selection
             ↓
        Scaling
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
        ```
        """
    )

    st.stop()


# ==========================================================
# READ UPLOADED FILE
# ==========================================================

file_bytes = uploaded_file.getvalue()

file_name = uploaded_file.name


# ==========================================================
# FILE SIZE INFORMATION
# ==========================================================

file_size_mb = (
    len(file_bytes)
    / (1024 * 1024)
)

st.caption(
    f"📄 {file_name} | "
    f"{file_size_mb:.1f} MB"
)


# ==========================================================
# PROCESS DATASET
# ==========================================================

with st.spinner(
    "🤖 AI is analyzing your dataset..."
):

    data = process_dataset(
        file_bytes,
        file_name
    )


# ==========================================================
# PROCESSING ERROR
# ==========================================================

if data["error"] == "missing_features":

    st.error(
        "❌ Required model features are missing."
    )

    st.write(
        "The trained model requires these features:"
    )

    st.write(
        features
    )

    st.write(
        "Missing from your uploaded CSV:"
    )

    st.error(
        ", ".join(
            data["missing_features"]
        )
    )

    st.stop()


# ==========================================================
# EXTRACT RESULTS
# ==========================================================

df = data["df"]

X = data["X"]

results = data["results"]

anomalies = data["anomalies"]

chart_results = data["chart_results"]

total_records = data["total_records"]

normal_count = data["normal_count"]

anomaly_count = data["anomaly_count"]

anomaly_rate = data["anomaly_rate"]


# ==========================================================
# DATA VALIDATION
# ==========================================================

st.header(
    "🔍 Data V
