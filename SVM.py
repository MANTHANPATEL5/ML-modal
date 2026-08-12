import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Weather SVM Prediction",
    page_icon="🌦️",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🌦️ Weather Prediction using SVM")

st.write(
    "Enter weather information to predict Rain / No Rain."
)


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    return pd.read_csv("Weather.csv")


df = load_data()


# ==========================================================
# FEATURES
# ==========================================================

features = [
    "Pressure",
    "global_radiation",
    "temp_mean(c)",
    "temp_min(c)",
    "temp_max(c)",
    "Wind_Speed",
    "Wind_Bearing"
]

target = "normalized_label"


# ==========================================================
# DATA
# ==========================================================

X = df[features].copy()

y = df[target].copy()


# ==========================================================
# MISSING VALUES
# ==========================================================

X = X.fillna(X.mean())


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
# SCALING
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================================
# TRAIN SVM
# ==========================================================

@st.cache_resource
def train_model(X_train_scaled, y_train):

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    return model


with st.spinner("Training SVM model..."):

    model = train_model(
        X_train_scaled,
        y_train
    )


# ==========================================================
# MODEL ACCURACY
# ==========================================================

y_pred = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==========================================================
# ACCURACY
# ==========================================================

st.subheader("🎯 Model Accuracy")

st.success(
    f"SVM Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================================
# USER INPUT
# ==========================================================

st.subheader("🌡️ Enter Weather Data")

st.write(
    "Please enter all weather values."
)


col1, col2 = st.columns(2)


# ==========================================================
# LEFT COLUMN
# ==========================================================

with col1:

    pressure = st.number_input(
        "Pressure",
        min_value=0.0,
        value=None,
        placeholder="Enter Pressure"
    )


    global_radiation = st.number_input(
        "Global Radiation",
        min_value=0.0,
        value=None,
        placeholder="Enter Global Radiation"
    )


    temp_mean = st.number_input(
        "Mean Temperature (°C)",
        value=None,
        placeholder="Enter Mean Temperature"
    )


    temp_min = st.number_input(
        "Minimum Temperature (°C)",
        value=None,
        placeholder="Enter Minimum Temperature"
    )


# ==========================================================
# RIGHT COLUMN
# ==========================================================

with col2:

    temp_max = st.number_input(
        "Maximum Temperature (°C)",
        value=None,
        placeholder="Enter Maximum Temperature"
    )


    wind_speed = st.number_input(
        "Wind Speed",
        min_value=0.0,
        value=None,
        placeholder="Enter Wind Speed"
    )


    wind_bearing = st.number_input(
        "Wind Bearing",
        min_value=0.0,
        value=None,
        placeholder="Enter Wind Bearing"
    )


# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button(
    "🔮 Predict Weather",
    use_container_width=True
):

    # ======================================================
    # CHECK EMPTY INPUT
    # ======================================================

    if (
        pressure is None
        or global_radiation is None
        or temp_mean is None
        or temp_min is None
        or temp_max is None
        or wind_speed is None
        or wind_bearing is None
    ):

        st.warning(
            "⚠️ Please enter all weather values before prediction."
        )

        st.stop()


    # ======================================================
    # USER DATA
    # ======================================================

    user_data = pd.DataFrame(
        [[
            pressure,
            global_radiation,
            temp_mean,
            temp_min,
            temp_max,
            wind_speed,
            wind_bearing
        ]],
        columns=features
    )


    # ======================================================
    # SCALE USER DATA
    # ======================================================

    user_data_scaled = scaler.transform(
        user_data
    )


    # ======================================================
    # PREDICTION
    # ======================================================

    prediction = model.predict(
        user_data_scaled
    )

    predicted_class = int(
        prediction[0]
    )


    # ======================================================
    # WEATHER CLASS
    # ======================================================

    class_labels = {

        0: "No Rain",

        1: "Rain",

        2: "Cloudy",

        3: "Storm"

    }


    predicted_weather = class_labels.get(
        predicted_class,
        "Unknown"
    )


    # ======================================================
    # RAIN YES / NO
    # ======================================================

    if predicted_class in [1, 3]:

        rain_result = "YES"

    else:

        rain_result = "NO"


    # ======================================================
    # RESULT
    # ======================================================

    st.subheader("🔮 Prediction Result")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Weather",
            predicted_weather
        )


    with result_col2:

        st.metric(
            "Rain",
            rain_result
        )


    # ======================================================
    # RESULT MESSAGE
    # ======================================================

    if predicted_class == 0:

        st.success(
            "☀️ NO RAIN"
        )


    elif predicted_class == 1:

        st.error(
            "🌧️ RAIN"
        )


    elif predicted_class == 2:

        st.info(
            "☁️ CLOUDY"
        )


    elif predicted_class == 3:

        st.error(
            "⛈️ STORM"
        )