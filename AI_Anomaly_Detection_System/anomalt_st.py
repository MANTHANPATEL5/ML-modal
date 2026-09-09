import os
import io
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "anomaly_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_FILE = os.path.join(BASE_DIR, "feature_info.pkl")

DISPLAY_ROWS = 100
CHART_SAMPLE_SIZE = 5000


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    "<style>"
    ".main-title {"
    "text-align:center;"
    "font-size:42px;"
    "font-weight:700;"
    "margin-bottom:5px;"
    "}"
    ".subtitle {"
    "text-align:center;"
    "color:#777;"
    "font-size:18px;"
    "margin-bottom:25px;"
    "}"
    "div[data-testid='stMetric'] {"
    "border:1px solid #dddddd;"
    "padding:12px;"
    "border-radius:10px;"
    "}"
    "</style>",
    unsafe_allow_html=True
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    "<div class='main-title'>🚨 AI Anomaly Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>"
    "Detect unusual patterns using Artificial Intelligence "
    "and Isolation Forest"
    "</div>",
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
    os.path.basename(file)
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error("❌ Trained model files are missing.")

    st.write("The following files are required:")

    for file in missing_files:
        st.write("• " + file)

    st.warning(
        "Place the trained model files in the same folder "
        "as anomalt_st.py."
    )

    st.info("Required project structure:")

    st.code(
        "AI_Anomaly_Detection_System/\n"
        "│\n"
        "├── anomalt_st.py\n"
        "├── train_model.py\n"
        "├── anomaly_model.pkl\n"
        "├── scaler.pkl\n"
        "├── feature_info.pkl\n"
        "└── requirements.txt",
        language="text"
    )

    st.stop()


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_info = joblib.load(FEATURE_FILE)

    return model, scaler, feature_info


try:

    model, scaler, feature_info = load_model_files()

    features = feature_info["features"]

except Exception as e:

    st.error("❌ Unable to load trained model.")

    st.error(str(e))

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.success("✅ Model Loaded")

st.sidebar.write("**Algorithm:** Isolation Forest")

st.sidebar.write("**Detection:** Unsupervised")

st.sidebar.write(
    "**Features:** " + str(len(features))
)

st.sidebar.write(
    "**Trees:** " + str(model.n_estimators)
)

st.sidebar.markdown("---")

st.sidebar.info(
    "The model detects unusual patterns from numerical "
    "transaction features."
)


# ==========================================================
# PROCESS DATASET
# ==========================================================

@st.cache_data(show_spinner=False)
def process_dataset(file_bytes):

    df = pd.read_csv(
        io.BytesIO(file_bytes)
    )

    rows = len(df)

    columns = len(df.columns)

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

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

    X = df[features].copy()

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        X[column] = X[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = X[column].fillna(
            median_value
        )

    X_scaled = scaler.transform(X)

    predictions = model.predict(
        X_scaled
    )

    decision_scores = model.decision_function(
        X_scaled
    )

    anomaly_scores = -decision_scores

    results = df.copy()

    results["Anomaly Score"] = anomaly_scores

    results["Prediction"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )

    total_records = len(results)

    anomaly_count = int(
        (
            results["Prediction"] == "Anomaly"
        ).sum()
    )

    normal_count = (
        total_records - anomaly_count
    )

    anomaly_rate = 0

    if total_records > 0:

        anomaly_rate = (
            anomaly_count / total_records
        ) * 100

    anomalies = results[
        results["Prediction"] == "Anomaly"
    ].copy()

    anomalies = anomalies.sort_values(
        "Anomaly Score",
        ascending=False
    )

    if len(results) > CHART_SAMPLE_SIZE:

        chart_results = results[
            ["Anomaly Score"]
        ].sample(
            CHART_SAMPLE_SIZE,
            random_state=42
        )

    else:

        chart_results = results[
            ["Anomaly Score"]
        ]

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
    "Upload your CSV dataset",
    type=["csv"],
    help="Upload a CSV containing the features used by the model."
)


# ==========================================================
# NO FILE
# ==========================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a CSV file to start anomaly detection."
    )

    st.subheader("🔄 AI Machine Learning Pipeline")

    st.code(
        "CSV Dataset\n"
        "     ↓\n"
        "Data Validation\n"
        "     ↓\n"
        "Missing-Value Handling\n"
        "     ↓\n"
        "Numerical Conversion\n"
        "     ↓\n"
        "Scaling\n"
        "     ↓\n"
        "Feature Selection\n"
        "     ↓\n"
        "Isolation Forest\n"
        "     ↓\n"
        "Anomaly Score\n"
        "     ↓\n"
        "Normal / Anomaly\n"
        "     ↓\n"
        "AI Explanation\n"
        "     ↓\n"
        "Visualization\n"
        "     ↓\n"
        "Download Results",
        language="text"
    )

    st.stop()


# ==========================================================
# PROCESS DATA
# ==========================================================

file_bytes = uploaded_file.getvalue()

with st.spinner(
    "🤖 AI is analyzing your dataset..."
):

    data = process_dataset(
        file_bytes
    )


# ==========================================================
# MISSING FEATURE ERROR
# ==========================================================

if data["error"] == "missing_features":

    st.error(
        "❌ Required features are missing."
    )

    st.write("Missing features:")

    for feature in data["missing_features"]:

        st.write("• " + feature)

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
# SCORE DISTRIBUTION
# ==========================================================

st.header("📉 Anomaly Score Distribution")

st.caption(
    "The chart uses a sample of records for faster performance."
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
        "Showing the most suspicious records."
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

    max_index = min(
        anomaly_count - 1,
        DISPLAY_ROWS - 1
    )

    selected_index = st.number_input(
        "Select anomaly number",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1
    )

    selected_row = anomalies.iloc[
        int(selected_index)
    ]

    explanation_data = []

    for feature in features:

        value = float(
            selected_row[feature]
        )

        typical_value = float(
            X[feature].median()
        )

        std_value = float(
            X[feature].std()
        )

        if (
            std_value == 0
            or np.isnan(std_value)
        ):

            deviation = 0

        else:

            deviation = abs(
                value - typical_value
            ) / std_value

        explanation_data.append(
            {
                "Feature": feature,
                "Observed Value": value,
                "Typical Value": typical_value,
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

    top_features = explanation_df.head(5)

    st.subheader(
        "🔎 Most Unusual Features"
    )

    st.dataframe(
        top_features,
        use_container_width=True
    )

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

    explanation_text = (
        "🔴 ANOMALY DETECTED\n\n"
        "The Isolation Forest model identified "
        "this record as unusual.\n\n"
        "Most unusual feature: "
        + str(strongest_feature)
        + "\n\n"
        "Observed value: "
        + f"{strongest_value:.4f}"
        + "\n\n"
        "Typical value: "
        + f"{strongest_typical:.4f}"
        + "\n\n"
        "Relative deviation: "
        + f"{strongest_deviation:.2f}"
        + " standard deviations\n\n"
        "The record was flagged because its "
        "feature pattern differs substantially "
        "from the normal patterns learned by "
        "the Isolation Forest model."
    )

    st.error(
        explanation_text
    )


# ==========================================================
# FEATURE ANALYSIS
# ==========================================================

st.header("📊 Feature Analysis")

if anomaly_count > 0:

    normal_data = results[
        results["Prediction"] == "Normal"
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
            anomaly_mean - normal_mean
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
        "Features with the largest difference "
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
    "Only the first 100 records are displayed. "
    "The complete dataset can be downloaded."
)

st.dataframe(
    results.head(DISPLAY_ROWS),
    use_container_width=True,
    height=450
)


# ==========================================================
# DOWNLOAD
# ==========================================================

st.header("⬇️ Download Results")

csv_output = results.to_csv(
    index=False
).encode("utf-8")

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
