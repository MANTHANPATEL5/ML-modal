import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded Successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================================
# 2. SELECT FEATURES
# ==========================================================

X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

# ==========================================================
# 3. SCALE DATA
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# 4. K-MEANS CLUSTERING
# ==========================================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# ==========================================================
# 5. DISPLAY CLUSTER RESULTS
# ==========================================================

print("\nClustered Data:")
print(
    df[
        [
            "CustomerID",
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Cluster"
        ]
    ].head(20)
)

# ==========================================================
# 6. CLUSTER COUNTS
# ==========================================================

print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# ==========================================================
# 7. CLUSTER CENTERS
# ==========================================================

centers = scaler.inverse_transform(kmeans.cluster_centers_)

print("\nCluster Centers:")

for i, center in enumerate(centers):
    print(
        f"Cluster {i}: "
        f"Income = {center[0]:.2f} k$, "
        f"Spending Score = {center[1]:.2f}"
    )

# ==========================================================
# 8. VISUALIZE CLUSTERS
# ==========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=80
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering - Mall Customers")
plt.legend()

plt.show()

# ==========================================================
# 9. SAVE CLUSTERED DATASET
# ==========================================================

df.to_csv("Mall_Customers_Clustered.csv", index=False)

print("\nClustered dataset saved as:")
print("Mall_Customers_Clustered.csv")
