import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

df = pd.read_csv("treatment_outcome.csv")

print("Treatment Outcome Data")
print(df.head())

# Encode categorical columns
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

features = ["Age", "Gender", "BloodPressure", "Cholesterol"]
X = df[features]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Scale the features (KNN is distance based)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredictions on Test Set")
result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print(result.head(10))

print("\nModel Performance")
print("Accuracy  =", round(accuracy_score(y_test, y_pred), 3))
print("Precision =", round(precision_score(y_test, y_pred, pos_label="Good"), 3))
print("Recall    =", round(recall_score(y_test, y_pred, pos_label="Good"), 3))
print("F1-Score  =", round(f1_score(y_test, y_pred, pos_label="Good"), 3))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred, labels=["Good", "Bad"])
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Treatment Outcome (KNN)")
plt.colorbar()
plt.xticks([0, 1], ["Good", "Bad"])
plt.yticks([0, 1], ["Good", "Bad"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()
