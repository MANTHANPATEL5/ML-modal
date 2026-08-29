import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="DBSCAN Life Expectancy",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 DBSCAN Life Expectancy Clustering")

st.write(
    "Enter the values below to identify the DBSCAN cluster."
)


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("LifeExpectancy.csv")


# ==========================================================
# SELECT FEATURES
# ==========================================================

features = [
    "Life expectancy",
    "Adult Mortality",
    "GDP",
    "Schooling"
]

X = df[features].copy()


# ==========================================================
# REMOVE MISSING VALUES
# ==========================================================

X = X.dropna()


# ==========================================================
# SCALE DATA
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================================
# DBSCAN MODEL
# ==========================================================

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

clusters = dbscan.fit_predict(X_scaled)

X["Cluster"] = clusters


# ==========================================================
# USER INPUT
# ==========================================================

st.subheader("📝 Enter Country Values")


life_expectancy = st.number_input(
    "Life Expectancy",
    min_value=0.0,
    max_value=100.0,
    value=None,
    placeholder="Enter Life Expectancy",
    step=0.1
)


adult_mortality = st.number_input(
    "Adult Mortality",
    min_value=0.0,
    max_value=1000.0,
    value=None,
    placeholder="Enter Adult Mortality",
    step=1.0
)


gdp = st.number_input(
    "GDP",
    min_value=0.0,
    max_value=200000.0,
    value=None,
    placeholder="Enter GDP",
    step=100.0
)


schooling = st.number_input(
    "Schooling",
    min_value=0.0,
    max_value=30.0,
    value=None,
    placeholder="Enter Schooling",
    step=0.1
)


# ==========================================================
# FIND CLUSTER
# ==========================================================

if st.button("🔍 Find Cluster"):

    # ======================================================
    # CHECK USER INPUT
    # ======================================================

    if (
        life_expectancy is None
        or adult_mortality is None
        or gdp is None
        or schooling is None
    ):

        st.warning(
            "⚠️ Please enter all four values."
        )

        st.stop()


    # ======================================================
    # USER DATA
    # ======================================================

    user_data = np.array([[
        life_expectancy,
        adult_mortality,
        gdp,
        schooling
    ]])


    # ======================================================
    # SCALE USER DATA
    # ======================================================

    user_scaled = scaler.transform(user_data)


    # ======================================================
    # FIND NEAREST DATASET POINT
    # ======================================================

    distances = np.linalg.norm(
        X_scaled - user_scaled,
        axis=1
    )

    nearest_index = np.argmin(distances)


    # ======================================================
    # GET CLUSTER
    # ======================================================

    predicted_cluster = clusters[nearest_index]


    # ======================================================
    # PREDICTION RESULT
    # ======================================================

    st.subheader("🎯 Prediction Result")


    if predicted_cluster == -1:

        st.error("🔴 Noise / Outlier")

        result_cluster = "Noise / Outlier"

    else:

        st.success(
            f"🟢 The input belongs to "
            f"Cluster {predicted_cluster}"
        )

        result_cluster = f"Cluster {predicted_cluster}"


    # ======================================================
    # USER INPUT DETAILS
    # ======================================================

    st.subheader("📋 User Input")


    result = pd.DataFrame({
        "Feature": [
            "Life Expectancy",
            "Adult Mortality",
            "GDP",
            "Schooling",
            "Cluster"
        ],

        "Value": [
            life_expectancy,
            adult_mortality,
            gdp,
            schooling,
            result_cluster
        ]
    })


    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )


    # ======================================================
    # USER INPUT VISUALIZATION
    # ======================================================

    st.subheader("📊 User Input Visualization")


    fig, ax = plt.subplots(figsize=(8, 5))


    # ------------------------------------------------------
    # PLOT USER INPUT
    # ------------------------------------------------------

    ax.scatter(
        life_expectancy,
        schooling,
        s=150
    )


    # ------------------------------------------------------
    # LABEL
    # ------------------------------------------------------

    ax.annotate(
        result_cluster,
        (
            life_expectancy,
            schooling
        ),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=11
    )


    # ------------------------------------------------------
    # CHART SETTINGS
    # ------------------------------------------------------

    ax.set_xlabel(
        "Life Expectancy"
    )

    ax.set_ylabel(
        "Schooling"
    )

    ax.set_title(
        "DBSCAN - User Input Cluster"
    )

    ax.grid(True)


    # ------------------------------------------------------
    # DISPLAY CHART
    # ------------------------------------------------------

    st.pyplot(fig)


    # ======================================================
    # DOWNLOAD CHART
    # ======================================================

    chart_buffer = BytesIO()

    fig.savefig(
        chart_buffer,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    chart_buffer.seek(0)


    st.download_button(
        label="📥 Download Chart",
        data=chart_buffer,
        file_name="DBSCAN_User_Input_Chart.png",
        mime="image/png"
    )
