import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest


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

MAX_TRAIN_ROWS = 100000
MAX_CHART_ROWS = 5000
DISPLAY_ROWS = 100

DEFAULT_CONTAMINATION = 0.0017
DEFAULT_TREES = 100


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

st.sidebar.title("⚙️ AI Model Settings")

st.sidebar.info(
    "This system automatically selects useful numeric "
    "features and detects unusual patterns."
)


trees = st.sidebar.slider(
    "Number of Trees",
    min_value=50,
    max_value=200,
    value=DEFAULT_TREES,
    step=10
)


contamination = st.sidebar.slider(
    "Expected Anomaly Rate",
    min_value=0.0005,
    max_value=0.05,
    value=DEFAULT_CONTAMINATION,
    step=0.0005,
    format="%.4f"
)


st.sidebar.markdown("---")

st.sidebar.write(
    "**Algorithm:** Isolation Forest"
)

st.sidebar.write(
    "**Learning:** Unsupervised"
)

st.sidebar.write(
    "**Feature Selection:** Automatic"
)

st.sidebar.write(
    "**Scaling:** Not required"
)

st.sidebar.write(
    f"**Training Limit:** {MAX_TRAIN_ROWS:,} rows"
)


# ==========================================================
# HELPER: IDENTIFY TARGET / LABEL COLUMNS
# ==========================================================

def find_label_columns(df):
    """
    Identify columns that should not be used as model features.
    """

    possible_labels = [
        "class",
        "label",
        "target",
        "fraud",
        "is_fraud",
        "anomaly",
        "anomaly_label",
        "y"
    ]

    excluded = []

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if normalized in possible_labels:
            excluded.append(column)

    return excluded


# ==========================================================
# HELPER: AUTOMATIC FEATURE SELECTION
# ==========================================================

def select_features(df):
    """
    Automatically select numeric columns suitable for
    Isolation Forest.
    """

    label_columns = find_label_columns(df)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    features = [
        column
        for column in numeric_columns
        if column not in label_columns
    ]

    return features, label_columns


# ==========================================================
# HELPER: CLEAN FEATURES
# ==========================================================

def clean_features(df, features):
    """
    Clean numeric features.

    Handles:
    - infinity
    - missing values
    - constant columns
    """

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
            median_value = 0.0

        X[column] = X[column].fillna(
            median_value
        )

    # Remove constant columns
    variable_features = []

    for column in X.columns:

        if X[column].nunique(dropna=False) > 1:
            variable_features.append(column)

    X = X[variable_features]

    return X, variable_features


# ==========================================================
# MODEL TRAINING
# ==========================================================

@st.cache_resource(show_spinner=False)
def train_isolation_forest(
    X,
    trees,
    contamination
):

    # ------------------------------------------------------
    # Training sample
    # ------------------------------------------------------

    if len(X) > MAX_TRAIN_ROWS:

        train_data = X.sample(
            n=MAX_TRAIN_ROWS,
            random_state=42
        )

    else:

        train_data = X

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = IsolationForest(
        n_estimators=trees,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        max_samples="auto"
    )

    model.fit(train_data)

    return model


# ==========================================================
# ANOMALY EXPLANATION
# ==========================================================

def create_explanation(
    selected_row,
    normal_data,
    features
):

    explanation = []

    for feature in features:

        value = float(
            selected_row[feature]
        )

        median_value = float(
            normal_data[feature].median()
        )

        std_value = float(
            normal_data[feature].std()
        )

        if (
            std_value == 0
            or np.isnan(std_value)
            or np.isinf(std_value)
        ):

            deviation = 0.0

        else:

            deviation = abs(
                value - median_value
            ) / std_value

        explanation.append(
            {
                "Feature": feature,
                "Observed Value": value,
                "Typical Value": median_value,
                "Deviation": deviation
            }
        )

    explanation_df = pd.DataFrame(
        explanation
    )

    explanation_df = explanation_df.sort_values(
        "Deviation",
        ascending=False
    )

    return explanation_df.reset_index(drop=True)


# ==========================================================
# FEATURE ANALYSIS
# ==========================================================

def calculate_feature_analysis(
    normal_data,
    anomaly_data,
    features
):

    analysis = []

    for feature in features:

        normal_median = normal_data[
            feature
        ].median()

        anomaly_median = anomaly_data[
            feature
        ].median()

        difference = abs(
            anomaly_median
            - normal_median
        )

        normal_std = normal_data[
            feature
        ].std()

        if (
            normal_std is None
            or normal_std == 0
            or np.isnan(normal_std)
        ):

            standardized_difference = 0.0

        else:

            standardized_difference = (
                difference / normal_std
            )

        analysis.append(
            {
                "Feature": feature,
                "Difference": difference,
                "Standardized Difference":
                    standardized_difference
            }
        )

    result = pd.DataFrame(
        analysis
    )

    return result.sort_values(
        "Standardized Difference",
        ascending=False
    )


# ==========================================================
# CSV PROCESSING
# ==========================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3
)
def process_dataset(
    file_bytes,
    trees,
    contamination
):

    # ------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------

    df = pd.read_csv(
        io.BytesIO(file_bytes)
    )

    # ------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------

    if df.empty:

        return {
            "error": "empty"
        }

    rows = len(df)

    columns = len(df.columns)

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    # ------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------

    selected_features, label_columns = select_features(
        df
    )

    if len(selected_features) == 0:

        return {
            "error": "no_features"
        }

    # ------------------------------------------------------
    # Clean features
    # ------------------------------------------------------

    X, features = clean_features(
        df,
        selected_features
    )

    if len(features) == 0:

        return {
            "error": "no_variable_features"
        }

    # ------------------------------------------------------
    # Train model
    # ------------------------------------------------------

    model = train_isolation_forest(
        X,
        trees,
        contamination
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    predictions = model.predict(
        X
    )

    # ------------------------------------------------------
    # Isolation Forest score
    #
    # Higher value = more suspicious
    # ------------------------------------------------------

    decision_scores = model.decision_function(
        X
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
    # Sort anomalies
    # ------------------------------------------------------

    anomalies = results[
        results["Prediction"] == "Anomaly"
    ].copy()

    anomalies = anomalies.sort_values(
        "Anomaly Score",
        ascending=False
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
    )

    # ------------------------------------------------------
    # Chart sample
    # ------------------------------------------------------

    if len(results) > MAX_CHART_ROWS:

        chart_results = results.sample(
            MAX_CHART_ROWS,
            random_state=42
        )

    else:

        chart_results = results.copy()

    chart_scores = chart_results[
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
        "features": features,
        "label_columns": label_columns,
        "results": results,
        "anomalies": anomalies,
        "chart_results": chart_scores,
        "model": model,
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
# UPLOAD DATASET
# ==========================================================

st.header("📁 Upload CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help=(
        "Credit card transactions, IoT data, "
        "business transactions, sensor data, etc."
    )
)


# ==========================================================
# NO DATASET
# ==========================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a CSV dataset to start AI anomaly detection."
    )

    st.markdown(
        """
        ### 🧠 AI Machine Learning Pipeline

        ```text
        CSV Dataset
             ↓
        Data Validation
             ↓
        Automatic Numeric Feature Selection
             ↓
        Missing-Value Handling
             ↓
        Isolation Forest Training
             ↓
        Anomaly Score
             ↓
        Normal / Anomaly
             ↓
        AI Explanation
             ↓
        Feature Analysis
             ↓
        Visualization
             ↓
        Download Results
        ```
        """
    )

    st.markdown("---")

    st.subheader("📌 Supported Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "💳 Credit Card Transactions"
        )

    with col2:
        st.info(
            "🏭 IoT / Sensor Data"
        )

    with col3:
        st.info(
            "🏢 Business Transactions"
        )

    st.stop()


# ==========================================================
# READ FILE
# ==========================================================

file_bytes = uploaded_file.getvalue()


# ==========================================================
# PROCESS DATA
# ==========================================================

with st.spinner(
    "🤖 AI is analyzing your dataset..."
):

    data = process_dataset(
        file_bytes,
        trees,
        contamination
    )


# ==========================================================
# HANDLE ERRORS
# ==========================================================

if data["error"] == "empty":

    st.error(
        "❌ The uploaded CSV file is empty."
    )

    st.stop()


if data["error"] == "no_features":

    st.error(
        "❌ No numeric features were found."
    )

    st.info(
        "The dataset must contain numeric columns "
        "that can be analyzed by Isolation Forest."
    )

    st.stop()


if data["error"] == "no_variable_features":

    st.error(
        "❌ No variable numeric features were found."
    )

    st.stop()


# ==========================================================
# EXTRACT DATA
# ==========================================================

df = data["df"]

X = data["X"]

features = data["features"]

label_columns = data["label_columns"]

results = data["results"]

anomalies = data["anomalies"]

chart_results = data["chart_results"]

model = data["model"]

total_records = data["total_records"]

normal_count = data["normal_count"]

anomaly_count = data["anomaly_count"]

anomaly_rate = data["anomaly_rate"]


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.success(
    "🤖 Model Ready"
)

st.sidebar.write(
    f"**Features Used:** {len(features)}"
)

st.sidebar.write(
    f"**Trees:** {model.n_estimators}"
)

st.sidebar.write(
    f"**Training Rows:** "
    f"{min(len(X), MAX_TRAIN_ROWS):,}"
)


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
# AUTOMATIC FEATURE SELECTION
# ==========================================================

st.header("🧠 Automatic Feature Selection")

st.success(
    f"{len(features)} numeric features automatically selected."
)

if label_columns:

    st.info(
        "Excluded label/target columns: "
        + ", ".join(
            map(str, label_columns)
        )
    )

st.dataframe(
    pd.DataFrame(
        {
            "Selected Features": features
        }
    ),
    use_container_width=True,
    height=250
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
    f"Chart uses up to {MAX_CHART_ROWS:,} "
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
        "Records are sorted from most suspicious "
        "to least suspicious."
    )

    display_anomalies = anomalies.head(
        DISPLAY_ROWS
    )

    st.dataframe(
        display_anomalies,
        use_container_width=True,
        height=450
    )


# ==========================================================
# AI ANOMALY EXPLANATION
# ==========================================================

st.header("🤖 AI Anomaly Explanation")

if anomaly_count > 0:

    max_selection = min(
        anomaly_count,
        DISPLAY_ROWS
    )

    selected_number = st.number_input(
        "Select anomaly",
        min_value=1,
        max_value=max_selection,
        value=1,
        step=1
    )

    selected_row = anomalies.iloc[
        selected_number - 1
    ]

    # ------------------------------------------------------
    # Normal reference data
    # ------------------------------------------------------

    normal_data = results[
        results["Prediction"] == "Normal"
    ]

    # ------------------------------------------------------
    # Explanation
    # ------------------------------------------------------

    explanation_df = create_explanation(
        selected_row,
        normal_data,
        features
    )

    top_features = explanation_df.head(5)

    st.subheader(
        "🔎 Most Unusual Features"
    )

    st.dataframe(
        top_features,
        use_container_width=True
    )

    # ------------------------------------------------------
    # Strongest feature
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
        "Deviation"
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

Why was it detected?

The selected transaction contains feature values
that differ substantially from the normal patterns
learned by the Isolation Forest model.
"""
    )


# ==========================================================
# FEATURE ANALYSIS
# ==========================================================

st.header("📊 Feature Analysis")

if anomaly_count > 0:

    normal_data = results[
        results["Prediction"] == "Normal"
    ]

    feature_analysis = calculate_feature_analysis(
        normal_data,
        anomalies,
        features
    )

    st.caption(
        "Features are ranked according to how strongly "
        "their anomalous values differ from normal records."
    )

    chart_data = feature_analysis.head(
        15
    ).set_index(
        "Feature"
    )[
        ["Standardized Difference"]
    ]

    st.bar_chart(
        chart_data
    )

    st.subheader(
        "📋 Feature Difference Details"
    )

    st.dataframe(
        feature_analysis.head(15),
        use_container_width=True
    )


# ==========================================================
# SELECTED ANOMALY DETAILS
# ==========================================================

if anomaly_count > 0:

    st.header("🔬 Selected Anomaly Details")

    selected_details = selected_row.to_frame(
        name="Value"
    )

    st.dataframe(
        selected_details,
        use_container_width=True
    )


# ==========================================================
# COMPLETE RESULTS
# ==========================================================

st.header("📋 Complete Results")

st.caption(
    f"Showing the first {DISPLAY_ROWS:,} records. "
    "The download button contains the complete dataset."
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
).encode("utf-8")

st.download_button(
    label="⬇️ Download Complete Anomaly Results CSV",
    data=csv_output,
    file_name="anomaly_detection_results.csv",
    mime="text/csv"
)


# ==========================================================
# DOWNLOAD ANOMALIES ONLY
# ==========================================================

if anomaly_count > 0:

    anomaly_csv = anomalies.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="🚨 Download Anomalies Only",
        data=anomaly_csv,
        file_name="detected_anomalies.csv",
        mime="text/csv"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "🧠 AI Anomaly Detection System | "
    "Isolation Forest | "
    "Unsupervised Machine Learning | "
    "Automatic Feature Selection"
)
