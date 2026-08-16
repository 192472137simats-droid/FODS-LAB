import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("ecommerce_transactions.csv")

print("E-commerce Transaction Data")
print(df.head())

features = ["TotalSpent", "ItemsPurchased"]
X = df[features]

# K-Means clustering on spending and purchase behavior
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Segment"] = model.fit_predict(X)

print("\nNumber of Customers in Each Segment")
print(df["Segment"].value_counts().sort_index())

print("\nAverage Behavior of Each Segment")
print(df.groupby("Segment")[features].mean())

# Visualization of clusters
plt.figure(figsize=(7, 5))
for seg in sorted(df["Segment"].unique()):
    part = df[df["Segment"] == seg]
    plt.scatter(part["TotalSpent"], part["ItemsPurchased"], label="Segment " + str(seg))
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("Customer Segmentation (Spending vs Items Purchased)")
plt.xlabel("Total Spent")
plt.ylabel("Items Purchased")
plt.legend()
plt.show()
