import pandas as pd

property_data = pd.DataFrame({
    'PropertyID': [1,2,3,4,5],
    'Location': ['Downtown','Suburb','Downtown','Suburb','Uptown'],
    'Bedrooms': [3,5,2,6,4],
    'Area_sqft': [1200,2500,900,3000,1800],
    'Price': [250000,400000,180000,500000,320000]
})

# 1. Average listing price per location
print(property_data.groupby('Location')['Price'].mean())

# 2. Count of properties with more than 4 bedrooms
print((property_data['Bedrooms'] > 4).sum())

# 3. Property with the largest area
print(property_data.loc[property_data['Area_sqft'].idxmax()])
