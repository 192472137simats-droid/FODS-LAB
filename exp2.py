import numpy as np

# 3x3 matrix: rows = products, columns = price entries recorded
sales_data = np.array([
    [250, 260, 255],   # Product 1 prices
    [400, 420, 410],   # Product 2 prices
    [150, 155, 160]    # Product 3 prices
])

# Average price of all products sold (overall mean of the matrix)
overall_average_price = np.mean(sales_data)

# Average price per product (row-wise mean)
average_price_per_product = np.mean(sales_data, axis=1)

# Output
print("Sales Data Matrix:\n", sales_data)
print("\nAverage price per product (row-wise):", average_price_per_product)
print("\nOverall average price of all products sold:", overall_average_price)
