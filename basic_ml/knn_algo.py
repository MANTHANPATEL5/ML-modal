import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="KNN Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("❤️ KNN Heart Disease Prediction")

st.write(
    "K-Nearest Neighbors Algorithm"
)


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("Heart_Disease.csv")


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop(
    "target",
    axis=1
)

y = df["target"]


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================================
# K VALUE
# ==========================================================

# Fixed K value
k = 5


# ==========================================================
# KNN MODEL
# ==========================================================

model = KNeighborsClassifier(
    n_neighbors=k
)

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================================
# TEST PREDICTION
# ==========================================================

y_pred = model.predict(
    X_test_scaled
)


# ==========================================================
# MODEL ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==========================================================
# ACCURACY DISPLAY
# ==========================================================

st.subheader("🎯 Model Accuracy")

st.metric(
    "KNN Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ==========================================================
# USER INPUT
# ==========================================================

st.divider()

st.header("👤 Patient Information")

st.write(
    "Please enter all patient information before prediction."
)


# ==========================================================
# TWO COLUMNS
# ==========================================================

col1, col2 = st.columns(2)


# ==========================================================
# AGE
# ==========================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=None,
        placeholder="Enter age"
    )


# ==========================================================
# SEX
# ==========================================================

with col2:

    sex = st.selectbox(
        "Sex",
        [
            "Select",
            "Male (1)",
            "Female (0)"
        ]
    )


# ==========================================================
# CHEST PAIN TYPE
# ==========================================================

with col1:

    chest_pain = st.selectbox(
        "Chest Pain Type",
        [
            "Select",
            "Type 1",
            "Type 2",
            "Type 3",
            "Type 4"
        ]
    )


# ==========================================================
# RESTING BLOOD PRESSURE
# ==========================================================

with col2:

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=None,
        placeholder="Enter blood pressure"
    )


# ==========================================================
# CHOLESTEROL
# ==========================================================

with col1:

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=None,
        placeholder="Enter cholesterol"
    )


# ==========================================================
# FASTING BLOOD SUGAR
# ==========================================================

with col2:

    fasting_blood_sugar = st.selectbox(
        "Fasting Blood Sugar",
        [
            "Select",
            "No (0)",
            "Yes (1)"
        ]
    )


# ==========================================================
# RESTING ECG
# ==========================================================

with col1:

    resting_ecg = st.selectbox(
        "Resting ECG",
        [
            "Select",
            "0",
            "1",
            "2"
        ]
    )


# ==========================================================
# MAXIMUM HEART RATE
# ==========================================================

with col2:

    max_heart_rate = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=None,
        placeholder="Enter maximum heart rate"
    )


# ==========================================================
# EXERCISE ANGINA
# ==========================================================

with col1:

    exercise_angina = st.selectbox(
        "Exercise Angina",
        [
            "Select",
            "No (0)",
            "Yes (1)"
        ]
    )


# ==========================================================
# OLDPEAK
# ==========================================================

with col2:

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=None,
        placeholder="Enter oldpeak",
        step=0.1
    )


# ==========================================================
# ST SLOPE
# ==========================================================

with col1:

    st_slope = st.selectbox(
        "ST Slope",
        [
            "Select",
            "1",
            "2",
            "3"
        ]
    )


# ==========================================================
# PREDICT BUTTON
# ==========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Heart Disease",
    use_container_width=True
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    # ======================================================
    # VALIDATE USER INPUT
    # ======================================================

    if age is None:

        st.warning(
            "⚠️ Please enter Age."
        )

    elif sex == "Select":

        st.warning(
            "⚠️ Please select Sex."
        )

    elif chest_pain == "Select":

        st.warning(
            "⚠️ Please select Chest Pain Type."
        )

    elif resting_bp is None:

        st.warning(
            "⚠️ Please enter Resting Blood Pressure."
        )

    elif cholesterol is None:

        st.warning(
            "⚠️ Please enter Cholesterol."
        )

    elif fasting_blood_sugar == "Select":

        st.warning(
            "⚠️ Please select Fasting Blood Sugar."
        )

    elif resting_ecg == "Select":

        st.warning(
            "⚠️ Please select Resting ECG."
        )

    elif max_heart_rate is None:

        st.warning(
            "⚠️ Please enter Maximum Heart Rate."
        )

    elif exercise_angina == "Select":

        st.warning(
            "⚠️ Please select Exercise Angina."
        )

    elif oldpeak is None:

        st.warning(
            "⚠️ Please enter Oldpeak."
        )

    elif st_slope == "Select":

        st.warning(
            "⚠️ Please select ST Slope."
        )

    else:

        # ==================================================
        # CONVERT SEX
        # ==================================================

        if sex == "Male (1)":

            sex_value = 1

        else:

            sex_value = 0


        # ==================================================
        # CONVERT CHEST PAIN
        # ==================================================

        chest_pain_value = int(
            chest_pain.split()[-1]
        )


        # ==================================================
        # CONVERT FASTING BLOOD SUGAR
        # ==================================================

        if fasting_blood_sugar == "Yes (1)":

            fasting_value = 1

        else:

            fasting_value = 0


        # ==================================================
        # CONVERT RESTING ECG
        # ==================================================

        resting_ecg_value = int(
            resting_ecg
        )


        # ==================================================
        # CONVERT EXERCISE ANGINA
        # ==================================================

        if exercise_angina == "Yes (1)":

            exercise_value = 1

        else:

            exercise_value = 0


        # ==================================================
        # CONVERT ST SLOPE
        # ==================================================

        st_slope_value = int(
            st_slope
        )


        # ==================================================
        # CREATE USER DATA
        # ==================================================

        user_data = pd.DataFrame(
            [[
                age,
                sex_value,
                chest_pain_value,
                resting_bp,
                cholesterol,
                fasting_value,
                resting_ecg_value,
                max_heart_rate,
                exercise_value,
                oldpeak,
                st_slope_value
            ]],
            columns=X.columns
        )


        # ==================================================
        # SCALE USER DATA
        # ==================================================

        user_data_scaled = scaler.transform(
            user_data
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(
            user_data_scaled
        )


        # ==================================================
        # CONFIDENCE
        # ==================================================

        probability = model.predict_proba(
            user_data_scaled
        )

        confidence = max(
            probability[0]
        ) * 100


        # ==================================================
        # RESULT
        # ==================================================

        st.subheader(
            "🎯 Prediction Result"
        )


        if prediction[0] == 1:

            st.error(
                "❤️ Heart Disease: YES"
            )

        else:

            st.success(
                "💚 Heart Disease: NO"
            )


        st.write(
            f"Prediction Confidence: "
            f"**{confidence:.2f}%**"
        )
