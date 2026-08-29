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

y_pred = model.predict(X_test)

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
    value="",
    placeholder="Enter number of pregnancies"
)

glu = st.text_input(
    "Glucose",
    value="",
    placeholder="Enter glucose level"
)

bp = st.text_input(
    "Blood Pressure",
    value="",
    placeholder="Enter blood pressure"
)

skin = st.text_input(
    "Skin Thickness",
    value="",
    placeholder="Enter skin thickness"
)

insulin = st.text_input(
    "Insulin",
    value="",
    placeholder="Enter insulin level"
)

bmi = st.text_input(
    "BMI",
    value="",
    placeholder="Enter BMI"
)

dpf = st.text_input(
    "Diabetes Pedigree Function",
    value="",
    placeholder="Enter diabetes pedigree function"
)

age = st.text_input(
    "Age",
    value="",
    placeholder="Enter age"
)


# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button(
    "🔮 Predict",
    use_container_width=True
):

    # ======================================================
    # VALIDATE INPUT
    # ======================================================

    if preg.strip() == "":
        st.warning("⚠️ Please enter Pregnancies.")

    elif glu.strip() == "":
        st.warning("⚠️ Please enter Glucose.")

    elif bp.strip() == "":
        st.warning("⚠️ Please enter Blood Pressure.")

    elif skin.strip() == "":
        st.warning("⚠️ Please enter Skin Thickness.")

    elif insulin.strip() == "":
        st.warning("⚠️ Please enter Insulin.")

    elif bmi.strip() == "":
        st.warning("⚠️ Please enter BMI.")

    elif dpf.strip() == "":
        st.warning("⚠️ Please enter Diabetes Pedigree Function.")

    elif age.strip() == "":
        st.warning("⚠️ Please enter Age.")

    else:

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
