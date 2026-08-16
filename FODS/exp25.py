import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("customer_reviews.csv")

ratings = df["Rating"]

mean = ratings.mean()
std = ratings.std()
n = len(ratings)

confidence = 0.95

t_value = stats.t.ppf(
    1 - (1 - confidence) / 2,
    n - 1
)

margin_error = t_value * std / (n ** 0.5)

lower = mean - margin_error
upper = mean + margin_error

print("Average Rating =", round(mean, 2))

print("95% Confidence Interval:")
print("Lower =", round(lower, 2))
print("Upper =", round(upper, 2))

# Graph
plt.figure(figsize=(7, 5))

plt.hist(
    ratings,
    bins=5
)

plt.axvline(
    mean,
    linestyle="--",
    label="Mean Rating"
)

plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Customers")

plt.legend()

plt.show()
