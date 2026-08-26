import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="K-Means Customer Clustering",
    page_icon="👥",
    layout="centered"
)

st.title("👥 K-Means Customer Clustering")
st.write("Enter customer details and predict the customer cluster.")

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("Mall_Customers.csv")

# ==========================================================
# 2. CONVERT GENDER TO NUMERIC
# ==========================================================

df["Gender_Numeric"] = df["Gender"].map({
    "Male": 0,
    "Female": 1
})

# ==========================================================
# 3. SELECT FEATURES
# ==========================================================

features = [
    "Age",
    "Gender_Numeric",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

# ==========================================================
# 4. SCALE DATA
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# 5. TRAIN K-MEANS
# ==========================================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# ==========================================================
# 6. CLUSTER CUSTOMER TYPES
# ==========================================================

cluster_types = {
    0: "Low income, low spending",
    1: "High income, high spending",
    2: "Medium income, medium/high spending",
    3: "High income, low spending",
    4: "Low income, high spending"
}

# ==========================================================
# 7. SHOW CLUSTER TYPES
# ==========================================================

st.subheader("📊 Cluster Customer Types")

cluster_table = pd.DataFrame({
    "Cluster": [
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
        "Cluster 3",
        "Cluster 4"
    ],
    "Customer Type": [
        "Low income, low spending",
        "High income, high spending",
        "Medium income, medium/high spending",
        "High income, low spending",
        "Low income, high spending"
    ]
})

st.table(cluster_table)

# ==========================================================
# 8. USER INPUT
# ==========================================================

st.subheader("📝 Enter Customer Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    step=1,
    value=None,
    placeholder="Enter age"
)

gender = st.selectbox(
    "Gender",
    ["Select Gender", "Male", "Female"]
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=1.0,
    max_value=200.0,
    step=1.0,
    value=None,
    placeholder="Enter annual income"
)

spending = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    step=1,
    value=None,
    placeholder="Enter spending score"
)

# ==========================================================
# 9. PREDICTION
# ==========================================================

if st.button("🔍 Predict Cluster", use_container_width=True):

    # Check user input
    if age is None:
        st.warning("⚠️ Please enter Age.")

    elif gender == "Select Gender":
        st.warning("⚠️ Please select Gender.")

    elif income is None:
        st.warning("⚠️ Please enter Annual Income.")

    elif spending is None:
        st.warning("⚠️ Please enter Spending Score.")

    else:

        # Convert gender
        gender_value = 0 if gender == "Male" else 1

        # Create user input
        user_data = [[
            age,
            gender_value,
            income,
            spending
        ]]

        # Scale user input
        user_scaled = scaler.transform(user_data)

        # Predict cluster
        cluster = kmeans.predict(user_scaled)[0]

        # Get customer type
        customer_type = cluster_types[cluster]

        # ==================================================
        # RESULT
        # ==================================================

        st.success(
            f"🎯 Customer belongs to Cluster {cluster}"
        )

        st.subheader("👤 Customer Type")

        st.info(
            f"**Cluster {cluster}: {customer_type}**"
        )