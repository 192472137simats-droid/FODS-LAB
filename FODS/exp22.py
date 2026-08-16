import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("q22_blood_pressure.csv")

drug = df[df["Group"] == "Drug"]["Reduction"]
placebo = df[df["Group"] == "Placebo"]["Reduction"]

def confidence_interval(data):
    mean = data.mean()
    sem = stats.sem(data)
    interval = stats.t.interval(
        0.95,
        len(data) - 1,
        loc=mean,
        scale=sem
    )
    return mean, interval

drug_mean, drug_ci = confidence_interval(drug)
placebo_mean, placebo_ci = confidence_interval(placebo)

print("Drug Group")
print("Mean Reduction =", round(drug_mean, 2))
print("95% Confidence Interval =", drug_ci)

print("\nPlacebo Group")
print("Mean Reduction =", round(placebo_mean, 2))
print("95% Confidence Interval =", placebo_ci)

# Graph
groups = ["Drug", "Placebo"]
means = [drug_mean, placebo_mean]

plt.figure(figsize=(7, 5))
plt.bar(groups, means)

plt.title("Mean Blood Pressure Reduction")
plt.xlabel("Group")
plt.ylabel("Mean Reduction")

plt.show()
