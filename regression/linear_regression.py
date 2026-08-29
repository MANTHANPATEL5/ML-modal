import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("house_price_dataset_200_rows.csv")

# Features and Target
X = df.drop("Price_USD", axis=1)
y = df["Price_USD"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

# ---------------- Streamlit UI ----------------

st.title("🏠 House Price Prediction")

st.write("Enter House Details")

# User Inputs
area = st.text_input("Enter Area (sqft)")
bedrooms = st.text_input("Enter Number of Bedrooms")
bathrooms = st.text_input("Enter Number of Bathrooms")
age = st.text_input("Enter House Age (Years)")
garage = st.text_input("Enter Garage Spaces")
distance = st.text_input("Enter Distance from City (km)")

if st.button("Predict Price"):

    if (
        area == "" or bedrooms == "" or bathrooms == "" or
        age == "" or garage == "" or distance == ""
    ):
        st.warning("Please enter all values.")
    else:
        try:
            input_data = pd.DataFrame({
                "Area_sqft": [float(area)],
                "Bedrooms": [float(bedrooms)],
                "Bathrooms": [float(bathrooms)],
                "Age_years": [float(age)],
                "Garage": [float(garage)],
                "Distance_km": [float(distance)]
            })

            prediction = model.predict(input_data)

            st.success(f"🏠 Predicted House Price: ${prediction[0]:,.2f}")

            st.subheader("Model Accuracy")
            st.write(f"R² Score: {accuracy:.4f}")

        except ValueError:
            st.error("Please enter numeric values only.")
