import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_csv("q21_age_bodyfat.csv")

print("DATA")
print(df)

# Mean
print("\nMean")
print(df.mean(numeric_only=True))

# Median
print("\nMedian")
print(df.median(numeric_only=True))

# Standard Deviation
print("\nStandard Deviation")
print(df.std(numeric_only=True))

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df)
plt.title("Boxplot of Age and Body Fat")
plt.show()

# Scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(df["Age"], df["Fat_Percent"])
plt.xlabel("Age")
plt.ylabel("Body Fat (%)")
plt.title("Age vs Body Fat")
plt.show()

# Q-Q plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

stats.probplot(df["Age"], dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot - Age")

stats.probplot(df["Fat_Percent"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot - Body Fat")

plt.tight_layout()
plt.show()
