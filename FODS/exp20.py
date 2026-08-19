#Question: Create a Python program that fulfills these requirements and helps your team gain insights  from the customer feedback data. 

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import string

df = pd.read_csv("data.csv")

stop_words = {
    "the", "and", "is", "to", "with", "a", "an",
    "of", "for", "in", "on", "at"
}

text = " ".join(df["feedback"]).lower()

text = text.translate(str.maketrans("", "", string.punctuation))

words = text.split()

filtered_words = [word for word in words if word not in stop_words]

frequency = Counter(filtered_words)

N = int(input("Enter the value of N: "))

top_words = frequency.most_common(N)

print("\nTop", N, "Most Frequent Words\n")

for word, count in top_words:
    print(word, ":", count)

words = [item[0] for item in top_words]
counts = [item[1] for item in top_words]

plt.figure(figsize=(8,5))
plt.bar(words, counts)
plt.title("Top Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
