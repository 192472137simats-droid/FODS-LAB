import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("mall_customers.csv")

print("Mall Customer Data")
print(df.head())

features = ["AnnualIncome", "SpendingScore"]
X = df[features]

# K-Means Clustering
model = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Segment"] = model.fit_predict(X)

print("\nEnter the details of the new customer")
income = float(input("Annual Income: "))
spending = float(input("Spending Score: "))

new_customer = [[income, spending]]
segment = model.predict(new_customer)[0]
print("\nThe new customer belongs to Segment", segment)

# Visualization
plt.figure(figsize=(7, 5))
plt.scatter(df["AnnualIncome"], df["SpendingScore"],
            c=df["Segment"], cmap="viridis")
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.scatter(income, spending, c="black", marker="*", s=250, label="New Customer")
plt.title("Customer Segments (K-Means)")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
