import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("customer_behavior.csv")

print("Customer Behavior Data")
print(df.head())

features = ["AnnualIncome", "SpendingScore", "VisitsPerMonth"]
X = df[features]

# K-Means clustering to segment customers
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Segment"] = model.fit_predict(X)

print("\nNumber of Customers in Each Segment")
print(df["Segment"].value_counts().sort_index())

print("\nAverage Characteristics of Each Segment")
print(df.groupby("Segment")[features].mean())

# Visualization
plt.figure(figsize=(7, 5))
plt.scatter(df["AnnualIncome"], df["SpendingScore"],
            c=df["Segment"], cmap="viridis")
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
