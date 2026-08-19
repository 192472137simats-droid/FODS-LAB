#Question: Develop a Python program to calculate the frequency distribution of words in the customer  reviews dataset? 
import pandas as pd
from collections import Counter

data = {
    "Review": [
        "Good product and excellent quality",
        "Excellent product and fast delivery",
        "Good quality and worth buying",
        "Fast delivery and good service",
        "Excellent quality and good packaging"
    ]
}

df = pd.DataFrame(data)

print("Customer Reviews")
print(df)

text = " ".join(df["Review"]).lower()

words = text.replace(".", "").replace(",", "").split()

frequency = Counter(words)

print("\nWord Frequency Distribution\n")

for word, count in frequency.items():
    print(word, ":", count)
