import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🏦 Random Forest Loan Approval")

st.write(
    "Enter applicant information to predict loan approval."
)


# ==========================================================
# LOAD DATASET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "loan_approval_dataset.csv"

df = pd.read_csv(DATA_PATH)


# ==========================================================
# CLEAN COLUMN NAMES
# ==========================================================

df.columns = df.columns.str.strip()


# ==========================================================
# REMOVE EMPTY ROWS
# ==========================================================

df = df.dropna()


# ==========================================================
# ENCODING
# ==========================================================

le_self = LabelEncoder()
le_education = LabelEncoder()
le_marital = LabelEncoder()
le_status = LabelEncoder()


df["Self_Employed"] = le_self.fit_transform(
    df["Self_Employed"].astype(str).str.strip()
)


df["Education"] = le_education.fit_transform(
    df["Education"].astype(str).str.strip()
)


df["Marital_Status"] = le_marital.fit_transform(
    df["Marital_Status"].astype(str).str.strip()
)


df["Loan_Status"] = le_status.fit_transform(
    df["Loan_Status"].astype(str).str.strip()
)


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop(
    "Loan_Status",
    axis=1
)

y = df["Loan_Status"]


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
# RANDOM FOREST MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
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
# MODEL ACCURACY DISPLAY
# ==========================================================

st.subheader("📊 Model Accuracy")


st.metric(
    "Random Forest Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ==========================================================
# USER INPUT
# ==========================================================

st.divider()

st.subheader("👤 User Input")


with st.form("loan_form"):

    col1, col2 = st.columns(2)


    # ======================================================
    # COLUMN 1
    # ======================================================

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=None,
            placeholder="Enter Age"
        )


        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=None,
            placeholder="Enter Annual Income"
        )


        credit_score = st.number_input(
            "Credit Score",
            min_value=0,
            max_value=900,
            value=None,
            placeholder="Enter Credit Score"
        )


        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=None,
            placeholder="Enter Loan Amount"
        )


        loan_term = st.number_input(
            "Loan Term (Months)",
            min_value=1,
            max_value=600,
            value=None,
            placeholder="Enter Loan Term"
        )


        employment_years = st.number_input(
            "Employment Years",
            min_value=0,
            max_value=60,
            value=None,
            placeholder="Enter Employment Years"
        )


        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=20,
            value=None,
            placeholder="Enter Existing Loans"
        )


    # ======================================================
    # COLUMN 2
    # ======================================================

    with col2:

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=20,
            value=None,
            placeholder="Enter Dependents"
        )


        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=None,
            placeholder="Enter Savings"
        )


        property_value = st.number_input(
            "Property Value",
            min_value=0.0,
            value=None,
            placeholder="Enter Property Value"
        )


        self_employed = st.selectbox(
            "Self Employed",
            ["Select"] + list(le_self.classes_)
        )


        education = st.selectbox(
            "Education",
            ["Select"] + list(le_education.classes_)
        )


        marital_status = st.selectbox(
            "Marital Status",
            ["Select"] + list(le_marital.classes_)
        )


    # ======================================================
    # SUBMIT BUTTON
    # ======================================================

    submit = st.form_submit_button(
        "🔍 Predict Loan Status",
        use_container_width=True
    )


# ==========================================================
# PREDICTION
# ==========================================================

if submit:

    # ======================================================
    # CHECK ALL INPUTS
    # ======================================================

    if age is None:

        st.warning("⚠️ Please enter Age.")
        st.stop()


    if annual_income is None:

        st.warning("⚠️ Please enter Annual Income.")
        st.stop()


    if credit_score is None:

        st.warning("⚠️ Please enter Credit Score.")
        st.stop()


    if loan_amount is None:

        st.warning("⚠️ Please enter Loan Amount.")
        st.stop()


    if loan_term is None:

        st.warning("⚠️ Please enter Loan Term.")
        st.stop()


    if employment_years is None:

        st.warning("⚠️ Please enter Employment Years.")
        st.stop()


    if existing_loans is None:

        st.warning("⚠️ Please enter Existing Loans.")
        st.stop()


    if dependents is None:

        st.warning("⚠️ Please enter Dependents.")
        st.stop()


    if savings is None:

        st.warning("⚠️ Please enter Savings.")
        st.stop()


    if property_value is None:

        st.warning("⚠️ Please enter Property Value.")
        st.stop()


    if self_employed == "Select":

        st.warning("⚠️ Please select Self Employed.")
        st.stop()


    if education == "Select":

        st.warning("⚠️ Please select Education.")
        st.stop()


    if marital_status == "Select":

        st.warning("⚠️ Please select Marital Status.")
        st.stop()


    # ======================================================
    # ENCODE USER INPUT
    # ======================================================

    self_encoded = le_self.transform(
        [self_employed]
    )[0]


    education_encoded = le_education.transform(
        [education]
    )[0]


    marital_encoded = le_marital.transform(
        [marital_status]
    )[0]


    # ======================================================
    # CREATE USER DATA
    # ======================================================

    new_data = pd.DataFrame({

        "Age": [age],

        "Annual_Income": [
            annual_income
        ],

        "Credit_Score": [
            credit_score
        ],

        "Loan_Amount": [
            loan_amount
        ],

        "Loan_Term_Months": [
            loan_term
        ],

        "Employment_Years": [
            employment_years
        ],

        "Existing_Loans": [
            existing_loans
        ],

        "Dependents": [
            dependents
        ],

        "Savings": [
            savings
        ],

        "Property_Value": [
            property_value
        ],

        "Self_Employed": [
            self_encoded
        ],

        "Education": [
            education_encoded
        ],

        "Marital_Status": [
            marital_encoded
        ]

    })


    # ======================================================
    # MATCH TRAINING COLUMN ORDER
    # ======================================================

    new_data = new_data[
        X.columns
    ]


    # ======================================================
    # PREDICTION
    # ======================================================

    prediction = model.predict(
        new_data
    )


    # ======================================================
    # PREDICTION PROBABILITY
    # ======================================================

    probability = model.predict_proba(
        new_data
    )


    confidence = max(
        probability[0]
    ) * 100


    # ======================================================
    # CONVERT RESULT BACK TO ORIGINAL LABEL
    # ======================================================

    result = le_status.inverse_transform(
        prediction
    )[0]


    # ======================================================
    # RESULT
    # ======================================================

    st.divider()

    st.subheader("🎯 Loan Prediction")


    if str(result).strip().lower() == "approved":

        st.success(
            "✅ LOAN APPROVED"
        )


    elif str(result).strip().lower() == "rejected":

        st.error(
            "❌ LOAN REJECTED"
        )


    else:

        st.info(
            f"Prediction: {result}"
        )


    # ======================================================
    # CONFIDENCE
    # ======================================================

    st.write(
        f"Prediction Confidence: **{confidence:.2f}%**"
    )
