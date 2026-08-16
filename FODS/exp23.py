import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("q23_ab_test.csv")

A = df[df["Design"] == "A"]["Converted"]
B = df[df["Design"] == "B"]["Converted"]

conversion_A = A.mean()
conversion_B = B.mean()

print("Design A Conversion Rate =", conversion_A * 100, "%")
print("Design B Conversion Rate =", conversion_B * 100, "%")

# Two sample t-test
t_stat, p_value = stats.ttest_ind(
    A,
    B,
    equal_var=False
)

print("\nt-statistic =", t_stat)
print("p-value =", p_value)

if p_value < 0.05:
    print("There is a statistically significant difference.")
else:
    print("There is no statistically significant difference.")

# Graph
plt.figure(figsize=(7, 5))

plt.bar(
    ["Design A", "Design B"],
    [conversion_A * 100, conversion_B * 100]
)

plt.title("Conversion Rate: Design A vs Design B")
plt.xlabel("Website Design")
plt.ylabel("Conversion Rate (%)")

plt.show()
