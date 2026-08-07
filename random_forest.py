import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Random Forest Loan Approval")


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("loan_approval_dataset.csv")

df.columns = df.columns.str.strip()

df = df.dropna()


# ==========================================
# ENCODING
# ==========================================

le_self = LabelEncoder()
le_education = LabelEncoder()
le_marital = LabelEncoder()
le_status = LabelEncoder()

df["Self_Employed"] = le_self.fit_transform(
    df["Self_Employed"]
)

df["Education"] = le_education.fit_transform(
    df["Education"]
)

df["Marital_Status"] = le_marital.fit_transform(
    df["Marital_Status"]
)

df["Loan_Status"] = le_status.fit_transform(
    df["Loan_Status"]
)


# ==========================================
# X AND Y
# ==========================================

X = df.drop(
    "Loan_Status",
    axis=1
)

y = df["Loan_Status"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# MODEL ACCURACY
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


st.subheader("📊 Model Accuracy")

st.metric(
    "Random Forest Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ==========================================
# USER INPUT
# ==========================================

st.subheader("👤 User Input")


with st.form("loan_form"):

    col1, col2 = st.columns(2)

    # ======================================
    # COLUMN 1
    # ======================================

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


    # ======================================
    # COLUMN 2
    # ======================================

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


    # ======================================
    # BUTTON
    # ======================================

    submit = st.form_submit_button(
        "🔍 Predict Loan Status",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if submit:

    # ======================================
    # CHECK BLANK INPUT
    # ======================================

    if (
        age is None
        or annual_income is None
        or credit_score is None
        or loan_amount is None
        or loan_term is None
        or employment_years is None
        or existing_loans is None
        or dependents is None
        or savings is None
        or property_value is None
        or self_employed == "Select"
        or education == "Select"
        or marital_status == "Select"
    ):

        st.warning(
            "⚠️ Please fill all fields."
        )

        st.stop()


    # ======================================
    # ENCODE USER INPUT
    # ======================================

    self_encoded = le_self.transform(
        [self_employed]
    )[0]

    education_encoded = le_education.transform(
        [education]
    )[0]

    marital_encoded = le_marital.transform(
        [marital_status]
    )[0]


    # ======================================
    # CREATE USER DATA
    # ======================================

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


    # ======================================
    # MATCH COLUMN ORDER
    # ======================================

    new_data = new_data[X.columns]


    # ======================================
    # PREDICTION
    # ======================================

    prediction = model.predict(
        new_data
    )


    # ======================================
    # RESULT
    # ======================================

    result = le_status.inverse_transform(
        prediction
    )[0]


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