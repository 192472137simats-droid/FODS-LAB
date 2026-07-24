import pandas as pd

data = {
    "City": ["Chennai", "Chennai", "Chennai",
             "Hyderabad", "Hyderabad", "Hyderabad",
             "Bangalore", "Bangalore", "Bangalore"],
    "Temperature": [34, 36, 35, 30, 32, 31, 26, 27, 26]
}

df = pd.DataFrame(data)

print("Temperature Data")
print(df)

stats = df.groupby("City")["Temperature"].agg(["mean", "std", "max", "min"])

stats["Range"] = stats["max"] - stats["min"]

print("\nTemperature Statistics")
print(stats)

highest_range = stats["Range"].idxmax()
most_consistent = stats["std"].idxmin()

print("\nMean Temperature for Each City")
print(stats["mean"])

print("\nStandard Deviation for Each City")
print(stats["std"])

print("\nCity with Highest Temperature Range:", highest_range)

print("City with Most Consistent Temperature:", most_consistent)
