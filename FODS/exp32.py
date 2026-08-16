import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("housing_prices.csv")

print("Housing Data")
print(df.head())

features = ["Area", "Bedrooms", "Age"]
X = df[features]
y = df["Price"]

# Linear Regression
model = LinearRegression()
model.fit(X, y)

print("\nEnter the features of the new house")
area = float(input("Area (sq ft): "))
bedrooms = float(input("Number of bedrooms: "))
age = float(input("Age of house (years): "))

new_house = [[area, bedrooms, age]]
predicted_price = model.predict(new_house)[0]

print("\nPredicted House Price = {:.2f}".format(predicted_price))

# Visualization - Area vs Price with regression trend
plt.figure(figsize=(7, 5))
plt.scatter(df["Area"], df["Price"], label="Actual")
plt.scatter(area, predicted_price, color="red", s=120, label="New House")
plt.title("Area vs Price")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.legend()
plt.show()
