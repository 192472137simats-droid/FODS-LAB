#Scenario: Decision Tree for Iris Flower Classification 
#You are analyzing the famous Iris flower dataset to classify iris flowers into three species based on  their sepal and petal dimensions. You want to use a Decision Tree classifier to accomplish this task.  Write a Python program that loads the Iris dataset from scikit-learn, and allows the user to input the  sepal length, sepal width, petal length, and petal width of a new flower. The program should then use  the Decision Tree classifier to predict the species of the new flower. 

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
