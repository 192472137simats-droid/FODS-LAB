import numpy as np

# 4x4 matrix: rows = students, columns = Math, Science, English, History
student_scores = np.array([
    [85, 90, 78, 92],
    [70, 88, 95, 60],
    [75, 82, 89, 91],
    [95, 70, 60, 85]
])

subjects = ["Math", "Science", "English", "History"]

# Average score for each subject (column-wise mean)
subject_averages = np.mean(student_scores, axis=0)

# Subject with highest average
best_subject_index = np.argmax(subject_averages)
best_subject = subjects[best_subject_index]

# Output
print("Student Scores Matrix:\n", student_scores)
print("\nAverage score per subject:")
for subj, avg in zip(subjects, subject_averages):
    print(f"{subj}: {avg}")

print(f"\nSubject with highest average score: {best_subject} ({subject_averages[best_subject_index]})")
