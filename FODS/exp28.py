# Classification and Regression Trees (CART) for Car Price Prediction 
#You are working for a car dealership, and you want to predict the price of used cars based on various  features such as the car's mileage, age, brand, and engine type. You have collected a dataset of used  cars with their respective prices. 
#Write a Python program that loads the car dataset and allows the user to input the features of a new car  they want to sell. The program should use the Classification and Regression Trees (CART) algorithm  from scikit-learn to predict the price of the new car based on the input features. The CART algorithm  will create a tree-based model that will split the data into subsets based on the chosen features and their  values, leading to a decision path that eventually predicts the price of the car. The program should  output the predicted price and display the decision path (the sequence of conditions leading to the  prediction) for the new car. 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text

df = pd.read_csv("car_price_data.csv")

print("Used Car Data")
print(df.head())

# Encode categorical features
brand_map = {name: i for i, name in enumerate(sorted(df["Brand"].unique()))}
engine_map = {name: i for i, name in enumerate(sorted(df["EngineType"].unique()))}

df["BrandCode"] = df["Brand"].map(brand_map)
df["EngineCode"] = df["EngineType"].map(engine_map)

features = ["Mileage", "Age", "BrandCode", "EngineCode"]
X = df[features]
y = df["Price"]

# CART algorithm (Regression Tree)
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X, y)

print("\nAvailable Brands:", list(brand_map.keys()))
print("Available Engine Types:", list(engine_map.keys()))

mileage = float(input("\nEnter mileage: "))
age = float(input("Enter age of car (years): "))
brand = input("Enter brand: ")
engine = input("Enter engine type: ")

new_car = [[mileage, age,
            brand_map.get(brand, 0),
            engine_map.get(engine, 0)]]

predicted_price = model.predict(new_car)[0]
print("\nPredicted Price = {:.2f}".format(predicted_price))

# Decision path (sequence of conditions leading to the prediction)
print("\nDecision Path:")
print(export_text(model, feature_names=features))

# Visualize the tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=features, filled=True, rounded=True, fontsize=8)
plt.title("CART Decision Tree - Car Price Prediction")
plt.show()
