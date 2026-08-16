import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("house_size_price.csv")

print("House Size and Price Data")
print(df.head())

# Bivariate analysis
correlation = df["Size"].corr(df["Price"])
print("\nCorrelation between Size and Price =", round(correlation, 3))

X = df[["Size"]]
y = df["Price"]

# Linear Regression model
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\nModel Coefficient (slope) =", round(model.coef_[0], 2))
print("Model Intercept =", round(model.intercept_, 2))

# Evaluate performance
print("\nModel Performance")
print("R-squared =", round(r2_score(y, y_pred), 3))
print("Mean Squared Error =", round(mean_squared_error(y, y_pred), 2))

# Visualization - scatter with regression line
plt.figure(figsize=(7, 5))
plt.scatter(df["Size"], df["Price"], label="Actual Data")
plt.plot(df["Size"], y_pred, color="red", label="Regression Line")
plt.title("Bivariate Analysis: House Size vs Price")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price")
plt.legend()
plt.show()
