import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

df = pd.read_csv("model_eval_data.csv")

print("Dataset")
print(df.head())
print("\nAvailable Columns:", list(df.columns))

feature_input = input("\nEnter feature names (comma separated): ")
target_input = input("Enter target variable name: ")

features = [f.strip() for f in feature_input.split(",")]
target = target_input.strip()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train a model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation metrics
print("\nEvaluation Metrics")
print("Accuracy  =", round(accuracy_score(y_test, y_pred), 3))
print("Precision =", round(precision_score(y_test, y_pred), 3))
print("Recall    =", round(recall_score(y_test, y_pred), 3))
print("F1-Score  =", round(f1_score(y_test, y_pred), 3))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Greens")
plt.title("Confusion Matrix - Model Evaluation")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()
