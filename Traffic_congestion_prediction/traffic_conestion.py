import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚦",
    layout="centered"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🚦 Traffic Congestion Prediction")

st.write(
    "Enter the traffic conditions below to predict "
    "the traffic congestion level."
)


# ==========================================================
# FILES
# ==========================================================

MODEL_FILE = "traffic_congestion_model.pkl"
DATASET_FILE = "traffic_data.csv"


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model_package = joblib.load(MODEL_FILE)

    model = model_package["model"]
    features = model_package["features"]
    classes = model_package["classes"]

    model_accuracy = model_package.get(
        "accuracy",
        None
    )

except FileNotFoundError:

    st.error(
        "❌ traffic_congestion_model.pkl was not found."
    )

    st.info(
        "Please run traffic.py first."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.stop()


# ==========================================================
# CHECK FEATURES
# ==========================================================

if len(features) != 10:

    st.error(
        f"❌ The loaded model contains {len(features)} "
        "features. Please train the model using the "
        "Top 10 feature version."
    )

    st.stop()


# ==========================================================
# LOAD DATASET
# Used only for min/max validation
# ==========================================================

try:

    df = pd.read_csv(DATASET_FILE)

except FileNotFoundError:

    st.error(
        "❌ traffic_data.csv was not found."
    )

    st.info(
        "Keep traffic_data.csv in the same folder "
        "as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading traffic_data.csv: {e}"
    )

    st.stop()


# ==========================================================
# FIND MINIMUM AND MAXIMUM VALUES
# ==========================================================

feature_ranges = {}

for feature in features:

    if feature not in df.columns:

        st.error(
            f"❌ Feature '{feature}' is not present "
            "in traffic_data.csv."
        )

        st.stop()


    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).dropna()


    if len(values) == 0:

        st.error(
            f"❌ No numeric values found for "
            f"'{feature}'."
        )

        st.stop()


    feature_ranges[feature] = {

        "min": float(values.min()),

        "max": float(values.max())

    }


# ==========================================================
# TRAFFIC INFORMATION
# ==========================================================

st.subheader("📊 Traffic Information")

st.write(
    "Enter values for the 10 important traffic features."
)


# ==========================================================
# USER INPUT
# ==========================================================

input_data = {}

invalid_features = []

empty_features = []


# ==========================================================
# INPUT BOXES
# ==========================================================

for feature in features:

    label = feature.replace(
        "_",
        " "
    ).title()


    minimum = feature_ranges[
        feature
    ]["min"]


    maximum = feature_ranges[
        feature
    ]["max"]


    # ------------------------------------------------------
    # USER INPUT
    # ------------------------------------------------------

    value = st.number_input(

        label=label,

        min_value=None,

        max_value=None,

        value=None,

        step=0.01,

        placeholder="Enter value",

        key=f"input_{feature}"

    )


    input_data[feature] = value


    # ------------------------------------------------------
    # EMPTY VALUE
    # ------------------------------------------------------

    if value is None:

        empty_features.append(
            feature
        )

        continue


    # ------------------------------------------------------
    # MINIMUM VALIDATION
    # ------------------------------------------------------

    if value < minimum:

        st.warning(
            f"⚠️ {label} should be at minimum "
            f"{minimum:.2f}."
        )

        invalid_features.append(
            feature
        )


    # ------------------------------------------------------
    # MAXIMUM VALIDATION
    # ------------------------------------------------------

    elif value > maximum:

        st.warning(
            f"⚠️ {label} should be at maximum "
            f"{maximum:.2f}."
        )

        invalid_features.append(
            feature
        )


# ==========================================================
# PREDICT BUTTON
# ==========================================================

st.write("")

predict_button = st.button(
    "🚦 Predict Traffic Congestion",
    use_container_width=True
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    # ======================================================
    # EMPTY CHECK
    # ======================================================

    if empty_features:

        st.error(
            "❌ Please enter all 10 traffic values."
        )

        st.stop()


    # ======================================================
    # INVALID VALUE CHECK
    # ======================================================

    if invalid_features:

        st.error(
            "❌ Please correct the values outside "
            "the allowed range."
        )

        st.stop()


    # ======================================================
    # CREATE DATAFRAME
    # ======================================================

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )


    try:

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(
            input_df
        )[0]


        # ==================================================
        # PROBABILITY
        # ==================================================

        probabilities = model.predict_proba(
            input_df
        )[0]


        # Highest probability
        confidence = (
            probabilities.max() * 100
        )


        # ==================================================
        # RESULT
        # ==================================================

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )


        # ==================================================
        # FREE-FLOW
        # ==================================================

        if prediction == "Free-flow":

            st.success(
                "🟢 FREE-FLOW"
            )

            st.write(
                "Traffic is moving normally "
                "with low congestion."
            )


        # ==================================================
        # MODERATE
        # ==================================================

        elif prediction == "Moderate":

            st.warning(
                "🟡 MODERATE"
            )

            st.write(
                "Traffic is moderately congested."
            )


        # ==================================================
        # HEAVY
        # ==================================================

        elif prediction == "Heavy":

            st.warning(
                "🟠 HEAVY"
            )

            st.write(
                "Traffic congestion is high."
            )


        # ==================================================
        # GRIDLOCK
        # ==================================================

        elif prediction == "Gridlock":

            st.error(
                "🔴 GRIDLOCK"
            )

            st.write(
                "Severe traffic congestion is predicted."
            )


        # ==================================================
        # OTHER
        # ==================================================

        else:

            st.info(
                f"🚦 {prediction}"
            )


        # ==================================================
        # CONFIDENCE
        # ==================================================

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # ==================================================
        # MODEL ACCURACY
        # ==================================================

        if model_accuracy is not None:

            st.write(
                f"**Model Test Accuracy:** "
                f"{model_accuracy * 100:.2f}%"
            )


        # ==================================================
        # PROBABILITIES
        # ==================================================

        st.subheader(
            "📈 Congestion Probabilities"
        )


        probability_df = pd.DataFrame({

            "Congestion Level":
                model.classes_,

            "Probability (%)":
                probabilities * 100

        })


        probability_df[
            "Probability (%)"
        ] = probability_df[
            "Probability (%)"
        ].round(2)


        st.dataframe(

            probability_df,

            use_container_width=True,

            hide_index=True

        )


        # ==================================================
        # CHART
        # ==================================================

        chart_df = probability_df.set_index(
            "Congestion Level"
        )


        st.bar_chart(
            chart_df["Probability (%)"]
        )


    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header(
        "🚦 Model Information"
    )


    st.write(
        "**Algorithm:** Random Forest Classifier"
    )


    st.write(
        "**Features Used:** 10"
    )


    if model_accuracy is not None:

        st.write(
            f"**Test Accuracy:** "
            f"{model_accuracy * 100:.2f}%"
        )


    st.write(
        "**Prediction Classes:** 4"
    )


    st.write("")

    st.write(
        "**Top 10 Features:**"
    )


    for feature in features:

        st.write(
            f"• {feature.replace('_', ' ').title()}"
        )


    st.write("")

    st.write(
        "**Classes:**"
    )


    for class_name in classes:

        st.write(
            f"• {class_name}"
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "🚦 Traffic Congestion Prediction | "
    "Random Forest"
)