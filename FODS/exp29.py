import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("iris.csv")

print("Iris Dataset")
print(df.head())

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[features]
y = df["species"]

# Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("\nEnter the measurements of the new flower")
sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

new_flower = [[sl, sw, pl, pw]]
prediction = model.predict(new_flower)[0]

print("\nPredicted Species =", prediction)

# Visualize the decision tree
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=features,
          class_names=sorted(df["species"].unique()),
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree - Iris Flower Classification")
plt.show()
