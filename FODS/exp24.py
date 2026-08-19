#Question: 
#write a Python program that allows the user to input the sample size, confidence level, and desired level of precision. 

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("rare_elements.csv")

sample_size = int(input("Enter sample size: "))
confidence = float(input("Enter confidence level (%): "))
precision = float(input("Enter desired precision: "))

data = df["Concentration"].sample(
    sample_size,
    random_state=42
)

mean = data.mean()
std = data.std()

alpha = 1 - confidence / 100

t_value = stats.t.ppf(
    1 - alpha / 2,
    sample_size - 1
)

margin_error = t_value * std / np.sqrt(sample_size)

lower = mean - margin_error
upper = mean + margin_error

print("\nPoint Estimate")
print("Sample Mean =", round(mean, 3))

print("\nConfidence Interval")
print("Lower Limit =", round(lower, 3))
print("Upper Limit =", round(upper, 3))

print("\nMargin of Error =", round(margin_error, 3))

if margin_error <= precision:
    print("Desired precision is achieved.")
else:
    print("Increase the sample size for better precision.")

# Graph
plt.figure(figsize=(8, 5))

plt.hist(
    data,
    bins=10
)

plt.axvline(
    mean,
    linestyle="--",
    label="Sample Mean"
)

plt.title("Rare Element Concentration")
plt.xlabel("Concentration")
plt.ylabel("Frequency")

plt.legend()
plt.show()
