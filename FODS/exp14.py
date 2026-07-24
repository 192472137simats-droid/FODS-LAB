import pandas as pd
import matplotlib.pyplot as plt

data = {
    "StudyTime": [1, 2, 3, 4, 5, 6, 7, 8],
    "ExamScore": [40, 50, 55, 65, 72, 80, 88, 95]
}

df = pd.DataFrame(data)

print("Student Data")
print(df)

correlation = df["StudyTime"].corr(df["ExamScore"])

print("\nCorrelation between Study Time and Exam Score:", correlation)

plt.figure(figsize=(6,4))
plt.scatter(df["StudyTime"], df["ExamScore"])
plt.title("Study Time vs Exam Score")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(df["StudyTime"], df["ExamScore"], marker='o')
plt.title("Study Time vs Exam Score")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()
