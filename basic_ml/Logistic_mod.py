import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🩺 Diabetes Prediction using Logistic Regression")

st.write(
    "Enter patient information to predict diabetes."
)


# ==========================================================
# LOAD DATASET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# Your actual GitHub filename contains a space before .csv
DATA_PATH = BASE_DIR / "diabetes .csv"

df = pd.read_csv(DATA_PATH)


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop(
    "Outcome",
    axis=1
)

y = df["Outcome"]


# ==========================================================
# SPLIT DATASET
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================================
# TRAIN MODEL
# ==========================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)


# ==========================================================
# MODEL ACCURACY
# ==========================================================

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==========================================================
# USER INPUT
# ==========================================================

st.header("👤 Patient Information")

preg = st.text_input(
    "Pregnancies",
    "2"
)

glu = st.text_input(
    "Glucose",
    "120"
)

bp = st.text_input(
    "Blood Pressure",
    "70"
)

skin = st.text_input(
    "Skin Thickness",
    "20"
)

insulin = st.text_input(
    "Insulin",
    "80"
)

bmi = st.text_input(
    "BMI",
    "25.5"
)

dpf = st.text_input(
    "Diabetes Pedigree Function",
    "0.45"
)

age = st.text_input(
    "Age",
    "35"
)


# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button(
    "🔮 Predict",
    use_container_width=True
):

    try:

        # ==================================================
        # CREATE PATIENT DATA
        # ==================================================

        patient = pd.DataFrame(
            [[
                int(preg),
                int(glu),
                int(bp),
                int(skin),
                int(insulin),
                float(bmi),
                float(dpf),
                int(age)
            ]],
            columns=X.columns
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(
            patient
        )


        # ==================================================
        # PREDICTION PROBABILITY
        # ==================================================

        probability = model.predict_proba(
            patient
        )


        # ==================================================
        # RESULT
        # ==================================================

        st.subheader("🎯 Prediction")


        if prediction[0] == 1:

            st.error(
                "🔴 Diabetic"
            )

        else:

            st.success(
                "🟢 Not Diabetic"
            )


        # ==================================================
        # PROBABILITY
        # ==================================================

        st.subheader("📊 Prediction Probability")

        st.write(
            f"Diabetic: **{probability[0][1] * 100:.2f}%**"
        )

        st.write(
            f"Not Diabetic: **{probability[0][0] * 100:.2f}%**"
        )


        # ==================================================
        # MODEL ACCURACY
        # ==================================================

        st.divider()

        st.metric(
            "🎯 Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    except ValueError:

        st.error(
            "⚠️ Please enter valid numeric values in all fields."
        )
