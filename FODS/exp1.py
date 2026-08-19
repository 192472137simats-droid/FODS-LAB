#1. Scenario: You are working on a project that involves analyzing student performance data for a class of 32 students. The data is stored in a NumPy array named student_scores, where each row represents a student and each column represents a different subject. The subjects are arranged in the  following order: Math, Science, English, and History. Your task is to calculate the average score for each subject and identify the subject with the highest average score. 
#Question: How would you use NumPy arrays to calculate the average score for each subject and determine the subject with the highest average score? Assume 4x4 matrix that stores marks of each student in given order. 

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
