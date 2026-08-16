import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("clinical_trial.csv")

print("Clinical Trial Data")
print(df.head())

control = df[df["Group"] == "Control"]["Score"]
treatment = df[df["Group"] == "Treatment"]["Score"]

print("\nControl Mean =", round(control.mean(), 2))
print("Treatment Mean =", round(treatment.mean(), 2))

# Hypothesis testing (two sample t-test)
t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

print("\nt-statistic =", round(t_stat, 4))
print("p-value =", p_value)

if p_value < 0.05:
    print("\nResult: The new treatment has a statistically significant effect.")
else:
    print("\nResult: No statistically significant effect of the treatment.")

# Visualization
plt.figure(figsize=(7, 5))
plt.hist(control, bins=10, alpha=0.6, label="Control")
plt.hist(treatment, bins=10, alpha=0.6, label="Treatment")
plt.axvline(control.mean(), linestyle="--", color="blue")
plt.axvline(treatment.mean(), linestyle="--", color="orange")
plt.title("Treatment vs Placebo (p-value = {:.4f})".format(p_value))
plt.xlabel("Recovery Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()
