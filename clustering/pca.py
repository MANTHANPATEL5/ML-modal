
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="PCA + K-Means Wine Clustering",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 PCA + K-Means Wine Clustering")

st.write(
    "Enter wine characteristics to predict its K-Means cluster."
)


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

# Get the folder where pca.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Wine dataset.csv should be in the same folder as pca.py
DATASET_PATH = os.path.join(BASE_DIR, "Wine dataset.csv")

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    st.error(
        "❌ Wine dataset.csv not found.\n\n"
        "Please place 'Wine dataset.csv' inside the "
        "'clustering' folder."
    )
    st.stop()


# ==========================================================
# 2. SELECT FEATURES
# ==========================================================

# Remove class because K-Means is unsupervised
if "class" in df.columns:
    X = df.drop("class", axis=1)
elif "Class" in df.columns:
    X = df.drop("Class", axis=1)
else:
    X = df.copy()


# Make sure all selected columns are numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows containing missing values
X = X.dropna()


# ==========================================================
# 3. STANDARDIZE DATA
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================================
# 4. APPLY PCA
# ==========================================================

# Reduce all features to 2 principal components
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# ==========================================================
# 5. K-MEANS CLUSTERING
# ==========================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_pca)


# ==========================================================
# 6. CREATE PCA DATAFRAME
# ==========================================================

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["Cluster"] = clusters


# ==========================================================
# 7. SILHOUETTE SCORE
# ==========================================================

silhouette = silhouette_score(
    X_pca,
    clusters
)


# ==========================================================
# 8. PCA + K-MEANS GRAPH
# ==========================================================

st.subheader("📊 PCA + K-Means Clustering")

fig, ax = plt.subplots(
    figsize=(10, 6)
)


# Plot clusters
for cluster in sorted(pca_df["Cluster"].unique()):

    points = pca_df[
        pca_df["Cluster"] == cluster
    ]

    ax.scatter(
        points["PC1"],
        points["PC2"],
        s=60,
        label=f"Cluster {cluster}"
    )


# ==========================================================
# CLUSTER CENTERS
# ==========================================================

centers = kmeans.cluster_centers_

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    s=200,
    marker="X",
    label="Cluster Centers"
)


# ==========================================================
# GRAPH SETTINGS
# ==========================================================

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

ax.set_title(
    "PCA + K-Means Clustering"
)

ax.legend()
ax.grid(True)

st.pyplot(fig)


# ==========================================================
# 9. SILHOUETTE SCORE
# ==========================================================

st.write(
    f"**Silhouette Score:** {silhouette:.4f}"
)


# ==========================================================
# 10. USER INPUT
# ==========================================================

st.subheader("🍷 New Wine Prediction")

st.write(
    "Enter all 13 wine feature values:"
)


with st.form("wine_prediction_form"):

    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        value=None,
        placeholder="Enter Alcohol"
    )

    malic_acid = st.number_input(
        "Malic acid",
        min_value=0.0,
        value=None,
        placeholder="Enter Malic acid"
    )

    ash = st.number_input(
        "Ash",
        min_value=0.0,
        value=None,
        placeholder="Enter Ash"
    )

    alcalinity = st.number_input(
        "Alcalinity of ash",
        min_value=0.0,
        value=None,
        placeholder="Enter Alcalinity of ash"
    )

    magnesium = st.number_input(
        "Magnesium",
        min_value=0.0,
        value=None,
        placeholder="Enter Magnesium"
    )

    total_phenols = st.number_input(
        "Total phenols",
        min_value=0.0,
        value=None,
        placeholder="Enter Total phenols"
    )

    flavanoids = st.number_input(
        "Flavanoids",
        min_value=0.0,
        value=None,
        placeholder="Enter Flavanoids"
    )

    nonflavanoid_phenols = st.number_input(
        "Nonflavanoid phenols",
        min_value=0.0,
        value=None,
        placeholder="Enter Nonflavanoid phenols"
    )

    proanthocyanins = st.number_input(
        "Proanthocyanins",
        min_value=0.0,
        value=None,
        placeholder="Enter Proanthocyanins"
    )

    color_intensity = st.number_input(
        "Color intensity",
        min_value=0.0,
        value=None,
        placeholder="Enter Color intensity"
    )

    hue = st.number_input(
        "Hue",
        min_value=0.0,
        value=None,
        placeholder="Enter Hue"
    )

    od280 = st.number_input(
        "OD280/OD315 of diluted wines",
        min_value=0.0,
        value=None,
        placeholder="Enter OD280/OD315"
    )

    proline = st.number_input(
        "Proline",
        min_value=0.0,
        value=None,
        placeholder="Enter Proline"
    )


    # ======================================================
    # PREDICT BUTTON
    # ======================================================

    submit = st.form_submit_button(
        "🔮 Predict Cluster"
    )


# ==========================================================
# 11. VALIDATE AND PREDICT
# ==========================================================

if submit:

    values = [
        alcohol,
        malic_acid,
        ash,
        alcalinity,
        magnesium,
        total_phenols,
        flavanoids,
        nonflavanoid_phenols,
        proanthocyanins,
        color_intensity,
        hue,
        od280,
        proline
    ]


    # ======================================================
    # CHECK EMPTY FIELDS
    # ======================================================

    if any(value is None for value in values):

        st.warning(
            "⚠️ Please enter all 13 wine feature values."
        )


    else:

        # ==================================================
        # CREATE NEW WINE DATA
        # ==================================================

        new_wine = pd.DataFrame(
            [[
                alcohol,
                malic_acid,
                ash,
                alcalinity,
                magnesium,
                total_phenols,
                flavanoids,
                nonflavanoid_phenols,
                proanthocyanins,
                color_intensity,
                hue,
                od280,
                proline
            ]],
            columns=X.columns
        )


        # ==================================================
        # STANDARDIZE NEW WINE
        # ==================================================

        new_wine_scaled = scaler.transform(
            new_wine
        )


        # ==================================================
        # APPLY PCA
        # ==================================================

        new_wine_pca = pca.transform(
            new_wine_scaled
        )


        # ==================================================
        # PREDICT CLUSTER
        # ==================================================

        predicted_cluster = kmeans.predict(
            new_wine_pca
        )[0]


        # ==================================================
        # DISPLAY PCA VALUES
        # ==================================================

        st.subheader("🔬 PCA Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "PC1",
                f"{new_wine_pca[0][0]:.4f}"
            )

        with col2:

            st.metric(
                "PC2",
                f"{new_wine_pca[0][1]:.4f}"
            )


        # ==================================================
        # FINAL PREDICTION
        # ==================================================

        st.subheader("🎯 Prediction Result")

        st.success(
            f"Predicted Wine Cluster: {predicted_cluster}"
        )

