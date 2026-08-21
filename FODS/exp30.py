#Write a Python program that allows the user to input the features of a new patient and the value of  k(number of neighbors).
# The program should use the KNN classifier from the scikit-learn library to predict whether the patient has the medical condition or not based on the input features. 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("patient_symptoms.csv")

print("Patient Symptom Data")
print(df.head())

features = ["Fever", "Cough", "Fatigue", "Pain"]
X = df[features]
y = df["Condition"]

print("\nEnter the symptoms of the new patient (0 = No, 1 = Yes)")
fever = int(input("Fever: "))
cough = int(input("Cough: "))
fatigue = int(input("Fatigue: "))
pain = int(input("Pain: "))
k = int(input("Enter value of k (number of neighbors): "))

# KNN Classifier
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X, y)

new_patient = [[fever, cough, fatigue, pain]]
prediction = model.predict(new_patient)[0]

if prediction == 1:
    print("\nPrediction: The patient HAS the medical condition.")
else:
    print("\nPrediction: The patient does NOT have the medical condition.")

# Visualization (Fever + Cough + Fatigue + Pain -> total symptoms)
df["TotalSymptoms"] = df[features].sum(axis=1)
plt.figure(figsize=(7, 5))
colors = df["Condition"].map({0: "green", 1: "red"})
plt.scatter(df.index, df["TotalSymptoms"], c=colors)
plt.axhline(sum(new_patient[0]), linestyle="--", label="New Patient")
plt.title("Patients by Total Symptoms (red = condition)")
plt.xlabel("Patient Index")
plt.ylabel("Total Symptoms")
plt.legend()
plt.show()
