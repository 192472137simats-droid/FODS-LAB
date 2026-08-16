import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("store_transactions.csv")

print("Store Transaction Data")
print(df.head())

features = ["TotalSpent", "VisitFrequency"]
X = df[features]

# K-Means clustering based on spending patterns
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Segment"] = model.fit_predict(X)

print("\nNumber of Customers in Each Segment")
print(df["Segment"].value_counts().sort_index())

print("\nAverage Spending Pattern of Each Segment")
print(df.groupby("Segment")[features].mean())

# Visualization
plt.figure(figsize=(7, 5))
plt.scatter(df["TotalSpent"], df["VisitFrequency"],
            c=df["Segment"], cmap="viridis")
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("Customer Segments by Spending Pattern")
plt.xlabel("Total Spent")
plt.ylabel("Visit Frequency")
plt.legend()
plt.show()
