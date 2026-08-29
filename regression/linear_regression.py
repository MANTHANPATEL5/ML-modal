import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# ==========================================================
# LOAD DATASET
# ==========================================================

# Get the folder where linear_regression.py is located
BASE_DIR = Path(__file__).resolve().parent

# CSV is in the same folder as linear_regression.py
DATA_PATH = BASE_DIR / "house_price_dataset_200_rows.csv"

df = pd.read_csv(DATA_PATH)


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop(
    "Price_USD",
    axis=1
)

y = df["Price_USD"]


# ==========================================================
# SPLIT DATA
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

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==========================================================
# MODEL PREDICTION
# ==========================================================

y_pred = model.predict(
    X_test
)


# ==========================================================
# MODEL ACCURACY
# ==========================================================

accuracy = r2_score(
    y_test,
    y_pred
)


# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("🏠 House Price Prediction")

st.write(
    "Enter House Details"
)


# ==========================================================
# USER INPUT
# ==========================================================

area = st.number_input(
    "Enter Area (sqft)",
    min_value=1.0,
    step=1.0,
    value=None,
    placeholder="Enter area"
)

bedrooms = st.number_input(
    "Enter Number of Bedrooms",
    min_value=1.0,
    step=1.0,
    value=None,
    placeholder="Enter bedrooms"
)

bathrooms = st.number_input(
    "Enter Number of Bathrooms",
    min_value=1.0,
    step=1.0,
    value=None,
    placeholder="Enter bathrooms"
)

age = st.number_input(
    "Enter House Age (Years)",
    min_value=0.0,
    step=1.0,
    value=None,
    placeholder="Enter house age"
)

garage = st.number_input(
    "Enter Garage Spaces",
    min_value=0.0,
    step=1.0,
    value=None,
    placeholder="Enter garage spaces"
)

distance = st.number_input(
    "Enter Distance from City (km)",
    min_value=0.0,
    step=0.1,
    value=None,
    placeholder="Enter distance"
)


# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button(
    "🔮 Predict Price",
    use_container_width=True
):

    # ======================================================
    # VALIDATE INPUT
    # ======================================================

    if area is None:

        st.warning("⚠️ Please enter Area.")

    elif bedrooms is None:

        st.warning("⚠️ Please enter Number of Bedrooms.")

    elif bathrooms is None:

        st.warning("⚠️ Please enter Number of Bathrooms.")

    elif age is None:

        st.warning("⚠️ Please enter House Age.")

    elif garage is None:

        st.warning("⚠️ Please enter Garage Spaces.")

    elif distance is None:

        st.warning("⚠️ Please enter Distance from City.")

    else:

        # ==================================================
        # CREATE INPUT DATA
        # ==================================================

        input_data = pd.DataFrame({
            "Area_sqft": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Age_years": [age],
            "Garage": [garage],
            "Distance_km": [distance]
        })


        # ==================================================
        # PREDICT PRICE
        # ==================================================

        prediction = model.predict(
            input_data
        )


        # ==================================================
        # SHOW RESULT
        # ==================================================

        st.success(
            f"🏠 Predicted House Price: "
            f"${prediction[0]:,.2f}"
        )


        # ==================================================
        # MODEL ACCURACY
        # ==================================================

        st.subheader("🎯 Model Accuracy")

        st.write(
            f"R² Score: **{accuracy:.4f}**"
        )
