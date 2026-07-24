import numpy as np

# house_data: each row = [bedrooms, square_footage, sale_price]
house_data = np.array([
    [3, 1500, 250000],
    [4, 1800, 300000],
    [5, 2200, 400000],
    [6, 2800, 500000],
    [2, 1200, 180000],
    [5, 2400, 420000],
    [7, 3200, 600000],
    [4, 1900, 310000]
])

# Step 1: Extract columns for readability
bedrooms = house_data[:, 0]
sale_price = house_data[:, 2]

# Step 2: Create a boolean mask for houses with more than 4 bedrooms
mask = bedrooms > 4

# Step 3: Filter sale prices using the mask
filtered_prices = sale_price[mask]

# Step 4: Calculate average sale price of filtered houses
average_price = np.mean(filtered_prices)

# Output
print("House Data (bedrooms, sqft, price):\n", house_data)
print("\nHouses with more than 4 bedrooms:\n", house_data[mask])
print("\nSale prices of these houses:", filtered_prices)
print("\nAverage sale price of houses with more than 4 bedrooms:", average_price)
