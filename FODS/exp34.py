import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("car_features.csv")

print("Car Features Data")
print(df.head())

features = ["EngineSize", "Horsepower", "FuelEfficiency"]
X = df[features]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Linear Regression modeling
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance")
print("R-squared =", round(r2_score(y_test, y_pred), 3))
print("Mean Squared Error =", round(mean_squared_error(y_test, y_pred), 2))

# Insights - most influential factors
print("\nInfluence of Each Feature on Car Price")
for feature, coef in zip(features, model.coef_):
    print(feature, "=", round(coef, 2))

# Visualization - feature coefficients
plt.figure(figsize=(7, 5))
plt.bar(features, model.coef_)
plt.title("Influence of Features on Car Price")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.axhline(0, color="black", linewidth=0.8)
plt.show()
