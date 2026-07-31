import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("diabetes .csv")



# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- STREAMLIT ----------------

st.set_page_config(page_title="Diabetes Prediction")

st.title("🩺 Diabetes Prediction using Logistic Regression")

# Text Inputs
preg = st.text_input("Pregnancies", "2")
glu = st.text_input("Glucose", "120")
bp = st.text_input("Blood Pressure", "70")
skin = st.text_input("Skin Thickness", "20")
insulin = st.text_input("Insulin", "80")
bmi = st.text_input("BMI", "25.5")
dpf = st.text_input("Diabetes Pedigree Function", "0.45")
age = st.text_input("Age", "35")

# Predict Button
if st.button("Predict"):

    try:
        patient = [[
            int(preg),
            int(glu),
            int(bp),
            int(skin),
            int(insulin),
            float(bmi),
            float(dpf),
            int(age)
        ]]

        prediction = model.predict(patient)
        probability = model.predict_proba(patient)

        st.subheader("Prediction")

        if prediction[0] == 1:
            st.error("🔴 Diabetic")
        else:
            st.success("🟢 Not Diabetic")

        st.write("### Prediction Probability")
        st.write(f"Diabetic : {probability[0][1]*100:.2f}%")
        st.write(f"Not Diabetic : {probability[0][0]*100:.2f}%")

        st.write("---")
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    except ValueError:
        st.error("Please enter valid numeric values in all fields.")