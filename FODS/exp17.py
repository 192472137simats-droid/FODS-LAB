#Question: Develop a code in python to find the frequency distribution of the ages of the customers who  have made a purchase in the past month. 
import pandas as pd

data = {
    "Customer": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "Age": [22, 25, 22, 30, 25, 35, 30, 22, 28, 25]
}

df = pd.DataFrame(data)

print("Customer Data")
print(df)

frequency = df["Age"].value_counts().sort_index()

print("\nFrequency Distribution of Customer Ages")
print(frequency)
