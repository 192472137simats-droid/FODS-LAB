#Question: Your task is to build a Python program that reads the stock data from a CSV file, calculates  the variability of stock prices, and provides insights into the stock's price movements. 

import pandas as pd

df = pd.read_csv("stock_data.csv")

print("Stock Data")
print(df)

mean_price = df["ClosingPrice"].mean()
variance = df["ClosingPrice"].var()
std_dev = df["ClosingPrice"].std()

print("\nMean Closing Price:", mean_price)
print("Variance:", variance)
print("Standard Deviation:", std_dev)

if std_dev > 5:
    print("\nInsight: Stock prices show HIGH variability.")
else:
    print("\nInsight: Stock prices show LOW variability.")
