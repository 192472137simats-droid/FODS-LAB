import numpy as np

# Sales data for Q1, Q2, Q3, Q4 (in $)
sales_data = np.array([50000, 62000, 58000, 75000])

# Total sales for the year
total_sales = np.sum(sales_data)

# Percentage increase from Q1 to Q4
percentage_increase = ((sales_data[3] - sales_data[0]) / sales_data[0]) * 100

print("Quarterly Sales:", sales_data)
print("Total sales for the year:", total_sales)
print(f"Percentage increase from Q1 to Q4: {percentage_increase:.2f}%")
