import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("customer_churn.csv")

print("Customer Churn Data")
print(df.head())

features = ["UsageMinutes", "ContractDuration", "MonthlyCharges"]
X = df[features]
y = df["Churn"]

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("\nEnter the details of the new customer")
usage = float(input("Usage Minutes: "))
contract = float(input("Contract Duration (months): "))
charges = float(input("Monthly Charges: "))

new_customer = [[usage, contract, charges]]
prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0][1]

if prediction == 1:
    print("\nPrediction: The customer is likely to CHURN.")
else:
    print("\nPrediction: The customer is likely to STAY.")

print("Churn Probability = {:.2f}%".format(probability * 100))

# Visualization - churn distribution
counts = df["Churn"].value_counts().sort_index()
plt.figure(figsize=(6, 5))
plt.bar(["Stay (0)", "Churn (1)"], counts.values)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.show()
