
import os
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

MODEL_FILE = "anomaly_model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_FILE = "feature_info.pkl"

# Number of rows shown in tables
DISPLAY_ROWS = 100

# Maximum points used for charts
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
# MODEL LOADING
# ==========================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_info = joblib.load(FEATURE_FILE)

    return model, scaler, feature_info


# ==========================================================
# CHECK MODEL FILES
# ==========================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_FILE
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error("❌ Trained model files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"• {file}")

    st.warning(
        "Run train_model.py first."
    )

    st.code(
        "python train_model.py",
        language="bash"
    )

    st.stop()


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model, scaler, feature_info = load_model_files()

    features = feature_info["features"]

except Exception as e:

    st.error("❌ Unable to load the trained model.")

    st.exception(e)

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.success("Model Loaded")

st.sidebar.write(
    "**Algorithm:** Isolation Forest"
)

st.sidebar.write(
    "**Detection:** Unsupervised"
)

st.sidebar.write(
    f"**Features:** {len(features)}"
)

st.sidebar.write(
    f"**Trees:** {model.n_estimators}"
)

st.sidebar.markdown("---")

st.sidebar.info(
    "The Class column is not used as a model input. "
    "The AI detects anomalies from transaction patterns."
)


# ==========================================================
# DATA PROCESSING FUNCTION
# ==========================================================

@st.cache_data(show_spinner=False)
def process_dataset(file_bytes, file_name):

    # ------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------

    from io import BytesIO

    df = pd.read_csv(
        BytesIO(file_bytes)
    )

    # ------------------------------------------------------
    # Validation
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
    # Required features
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
    # Feature dataframe
    # ------------------------------------------------------

    X = df[
        features
    ].copy()


    # ------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------

    for column in features:

        X[column] = X[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

        X[column] = X[column].fillna(
            X[column].median()
        )


    # ------------------------------------------------------
    # Scaling
    # ------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )


    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    predictions = model.predict(
        X_scaled
    )


    # ------------------------------------------------------
    # Anomaly score
    # ------------------------------------------------------

    decision_scores = model.decision_function(
        X_scaled
    )

    anomaly_scores = -decision_scores


    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    results = df.copy()

    results["Anomaly Score"] = anomaly_scores

    results["Prediction"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )


    # ------------------------------------------------------
    # Statistics
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

    anomaly_rate = (
        anomaly_count
        / total_records
        * 100
        if total_records > 0
        else 0
    )


    # ------------------------------------------------------
    # Top anomalies
    # ------------------------------------------------------

    anomalies = results[
        results["Prediction"]
        == "Anomaly"
    ].copy()

    anomalies = anomalies.sort_values(
        "Anomaly Score",
        ascending=False
    )


    # ------------------------------------------------------
    # Chart sampling
    # ------------------------------------------------------

    if len(results) > CHART_SAMPLE_SIZE:

        chart_results = results[
            ["Anomaly Score"]
        ].sample(
            CHART_SAMPLE_SIZE,
            random_state=42
        ).sort_values(
            "Anomaly Score"
        ).reset_index(
            drop=True
        )

    else:

        chart_results = results[
            ["Anomaly Score"]
        ].sort_values(
            "Anomaly Score"
        ).reset_index(
            drop=True
        )


    # ------------------------------------------------------
    # Return
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

st.header("📁 Upload CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload your creditcard.csv",
    type=["csv"],
    help="Upload the CSV used by the trained model."
)


# ==========================================================
# NO FILE
# ==========================================================

if uploaded_file is None:

    st.info(
        "👆 Upload your CSV file to start AI anomaly detection."
    )

    st.markdown(
        """
        ### 🔄 Machine Learning Pipeline

        ```text
        CSV Dataset
             ↓
        Data Validation
             ↓
        Missing-Value Handling
             ↓
        Scaling
             ↓
        Automatic Feature Selection
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
# PROCESS DATA
# ==========================================================

file_bytes = uploaded_file.getvalue()

file_name = uploaded_file.name


with st.spinner(
    "🤖 AI is analyzing the dataset... Please wait..."
):

    data = process_dataset(
        file_bytes,
        file_name
    )


# ==========================================================
# ERROR
# ==========================================================

if data["error"] == "missing_features":

    st.error(
        "❌ Required features are missing from the uploaded dataset."
    )

    st.write("Missing features:")

    st.write(
        data["missing_features"]
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

st.header("🔍 Data Validation")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        f"{data['rows']:,}"
    )

with col2:

    st.metric(
        "Columns",
        f"{data['columns']:,}"
    )

with col3:

    st.metric(
        "Missing Values",
        f"{data['missing_values']:,}"
    )

with col4:

    st.metric(
        "Duplicates",
        f"{data['duplicates']:,}"
    )


# ==========================================================
# DETECTION SUMMARY
# ==========================================================

st.header("📊 Detection Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Records",
        f"{total_records:,}"
    )

with col2:

    st.metric(
        "🟢 Normal",
        f"{normal_count:,}"
    )

with col3:

    st.metric(
        "🔴 Anomalies",
        f"{anomaly_count:,}"
    )

with col4:

    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )


# ==========================================================
# ANOMALY DISTRIBUTION
# ==========================================================

st.header("📈 Anomaly Distribution")

distribution = pd.DataFrame(
    {
        "Records": [
            normal_count,
            anomaly_count
        ]
    },
    index=[
        "Normal",
        "Anomaly"
    ]
)

st.bar_chart(
    distribution
)


# ==========================================================
# ANOMALY SCORE DISTRIBUTION
# ==========================================================

st.header("📉 Anomaly Score Distribution")

st.caption(
    f"Visualization uses up to {CHART_SAMPLE_SIZE:,} sampled "
    "records for better performance."
)

st.line_chart(
    chart_results
)


# ==========================================================
# DETECTED ANOMALIES
# ==========================================================

st.header("🔴 Detected Anomalies")

if anomaly_count == 0:

    st.success(
        "🎉 No anomalies were detected."
    )

else:

    st.warning(
        f"{anomaly_count:,} anomalies detected."
    )

    st.caption(
        f"Showing the top {min(DISPLAY_ROWS, anomaly_count):,} "
        "most suspicious records."
    )

    st.dataframe(
        anomalies.head(DISPLAY_ROWS),
        use_container_width=True,
        height=450
    )


# ==========================================================
# AI EXPLANATION
# ==========================================================

st.header("🤖 AI Anomaly Explanation")


if anomaly_count > 0:

    # ------------------------------------------------------
    # Select anomaly
    # ------------------------------------------------------

    selected_index = st.number_input(
        "Select anomaly number",
        min_value=0,
        max_value=min(
            anomaly_count - 1,
            DISPLAY_ROWS - 1
        ),
        value=0,
        step=1
    )


    selected_row = anomalies.iloc[
        selected_index
    ]


    # ------------------------------------------------------
    # Calculate explanation
    # ------------------------------------------------------

    explanation_data = []

    for feature in features:

        value = float(
            selected_row[feature]
        )

        median_value = float(
            X[feature].median()
        )

        std_value = float(
            X[feature].std()
        )

        if std_value == 0 or np.isnan(std_value):

            deviation = 0.0

        else:

            deviation = abs(
                value - median_value
            ) / std_value


        explanation_data.append(
            {
                "Feature": feature,
                "Observed Value": value,
                "Typical Value": median_value,
                "Relative Deviation": deviation
            }
        )


    explanation_df = pd.DataFrame(
        explanation_data
    )

    explanation_df = explanation_df.sort_values(
        "Relative Deviation",
        ascending=False
    ).reset_index(
        drop=True
    )


    # ------------------------------------------------------
    # Top unusual features
    # ------------------------------------------------------

    top_features = explanation_df.head(5)


    st.subheader(
        "🔎 Most Unusual Features"
    )

    st.dataframe(
        top_features,
        use_container_width=True
    )


    # ------------------------------------------------------
    # Main explanation
    # ------------------------------------------------------

    strongest = top_features.iloc[0]

    strongest_feature = strongest[
        "Feature"
    ]

    strongest_value = strongest[
        "Observed Value"
    ]

    strongest_typical = strongest[
        "Typical Value"
    ]

    strongest_deviation = strongest[
        "Relative Deviation"
    ]


    st.error(
        f"""
🔴 ANOMALY DETECTED

The Isolation Forest model identified this
record as unusual.

Most unusual feature:
{strongest_feature}

Observed value:
{strongest_value:.4f}

Typical value:
{strongest_typical:.4f}

Relative deviation:
{strongest_deviation:.2f} standard deviations

The record was flagged because its combination
of feature values differs substantially from the
normal patterns learned by the AI model.
"""
    )


# ==========================================================
# FEATURE ANALYSIS
# ==========================================================

st.header("📊 Feature Analysis")

if anomaly_count > 0:

    # Use only anomaly records for this analysis.
    # This avoids repeatedly calculating over all 284k rows.

    normal_data = results[
        results["Prediction"]
        == "Normal"
    ]


    feature_difference = []

    for feature in features:

        normal_mean = normal_data[
            feature
        ].mean()

        anomaly_mean = anomalies[
            feature
        ].mean()

        difference = abs(
            anomaly_mean
            - normal_mean
        )

        feature_difference.append(
            {
                "Feature": feature,
                "Difference": difference
            }
        )


    feature_analysis = pd.DataFrame(
        feature_difference
    )

    feature_analysis = feature_analysis.sort_values(
        "Difference",
        ascending=False
    )


    st.caption(
        "Top 15 features with the largest difference "
        "between normal and anomalous records."
    )

    st.bar_chart(
        feature_analysis.head(15).set_index(
            "Feature"
        )
    )


# ==========================================================
# COMPLETE RESULTS
# ==========================================================

st.header("📋 Complete Results")

st.caption(
    f"Showing first {DISPLAY_ROWS:,} records here. "
    "The complete dataset is available through the download button."
)

st.dataframe(
    results.head(DISPLAY_ROWS),
    use_container_width=True,
    height=450
)


# ==========================================================
# DOWNLOAD RESULTS
# ==========================================================

st.header("⬇️ Download Results")

csv_output = results.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="⬇️ Download Complete Anomaly Results CSV",
    data=csv_output,
    file_name="anomaly_detection_results.csv",
    mime="text/csv"
)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "🧠 AI Anomaly Detection System | "
    "Isolation Forest | "
    "Unsupervised Machine Learning"
)
