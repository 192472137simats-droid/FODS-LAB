#Question: Write a python program will take in a dataset containing daily temperature readings for each  city over a year and perform the following tasks: 
#1. Calculate the mean temperature for each city. 
#2. Calculate the standard deviation of temperature for each city. 
#3. Determine the city with the highest temperature range (difference between the highest and lowest  temperatures). 
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
