import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction")
st.write("Decision Tree Classifier using Gini Index")


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("loan_approval_dataset.csv")


# ==========================================
# ENCODING
# ==========================================

df["Self_Employed"] = df["Self_Employed"].map({
    "No": 0,
    "Yes": 1
})

df["Education"] = df["Education"].map({
    "High School": 0,
    "Graduate": 1,
    "Postgraduate": 2
})

df["Marital_Status"] = df["Marital_Status"].map({
    "Single": 0,
    "Married": 1
})

df["Loan_Status"] = df["Loan_Status"].map({
    "Rejected": 0,
    "Approved": 1
})


# ==========================================
# X AND Y
# ==========================================

X = df.drop("Loan_Status", axis=1)

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
# DECISION TREE MODEL
# ==========================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# MODEL ACCURACY
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==========================================
# DISPLAY ACCURACY
# ==========================================

st.subheader("📊 Model Accuracy")

st.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ==========================================
# USER INPUT
# ==========================================

st.subheader("👤 Enter Customer Details")

col1, col2 = st.columns(2)


# ==========================================
# COLUMN 1
# ==========================================

with col1:

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=100,
        value=None,
        placeholder="Enter age"
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=None,
        placeholder="Enter annual income"
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=None,
        placeholder="Enter credit score"
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=None,
        placeholder="Enter loan amount"
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [
            "Select Loan Term",
            12,
            24,
            36,
            48,
            60,
            72,
            84
        ]
    )

    employment_years = st.number_input(
        "Employment Years",
        min_value=0,
        max_value=50,
        value=None,
        placeholder="Enter employment years"
    )


# ==========================================
# COLUMN 2
# ==========================================

with col2:

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=None,
        placeholder="Enter existing loans"
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=None,
        placeholder="Enter dependents"
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=None,
        placeholder="Enter savings"
    )

    property_value = st.number_input(
        "Property Value",
        min_value=0.0,
        value=None,
        placeholder="Enter property value"
    )

    self_employed = st.selectbox(
        "Self Employed",
        [
            "Select",
            "No",
            "Yes"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "Select",
            "High School",
            "Graduate",
            "Postgraduate"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Select",
            "Single",
            "Married"
        ]
    )


# ==========================================
# SMALL BUTTON STYLE
# ==========================================

st.markdown(
    """
    <style>

    div.stButton > button {
        width: 180px;
        height: 42px;
        font-size: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Predict Loan"):

    # ======================================
    # CHECK EMPTY INPUT
    # ======================================

    if (
        age is None
        or annual_income is None
        or credit_score is None
        or loan_amount is None
        or loan_term == "Select Loan Term"
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
            "⚠️ Please fill in all details before prediction."
        )


    else:

        # ==================================
        # CONVERT USER INPUT
        # ==================================

        self_employed_value = {
            "No": 0,
            "Yes": 1
        }[self_employed]

        education_value = {
            "High School": 0,
            "Graduate": 1,
            "Postgraduate": 2
        }[education]

        marital_status_value = {
            "Single": 0,
            "Married": 1
        }[marital_status]


        # ==================================
        # CREATE USER DATA
        # ==================================

        user_data = pd.DataFrame([{

            "Age": age,

            "Annual_Income": annual_income,

            "Credit_Score": credit_score,

            "Loan_Amount": loan_amount,

            "Loan_Term_Months": loan_term,

            "Employment_Years": employment_years,

            "Existing_Loans": existing_loans,

            "Dependents": dependents,

            "Savings": savings,

            "Property_Value": property_value,

            "Self_Employed": self_employed_value,

            "Education": education_value,

            "Marital_Status": marital_status_value

        }])


        # ==================================
        # PREDICTION
        # ==================================

        prediction = model.predict(
            user_data
        )


        # ==================================
        # PREDICTION RESULT
        # ==================================

        st.subheader("📋 Prediction Result")

        if prediction[0] == 1:

            st.success(
                "🎉 LOAN APPROVED"
            )

        else:

            st.error(
                "❌ LOAN REJECTED"
            )
