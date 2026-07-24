import pandas as pd

order_data = pd.DataFrame({
    'CustomerID': [101, 102, 101, 103, 102, 101, 104],
    'OrderDate': pd.to_datetime(['2024-01-05','2024-01-07','2024-01-10',
                                  '2024-01-12','2024-01-15','2024-01-20','2024-01-25']),
    'Product': ['Pen','Notebook','Pen','Bag','Notebook','Bag','Pen'],
    'Quantity': [10, 5, 8, 2, 3, 4, 6]
})

# 1. Total orders per customer
print(order_data.groupby('CustomerID')['Product'].count())

# 2. Average order quantity per product
print(order_data.groupby('Product')['Quantity'].mean())

# 3. Earliest and latest order dates
print(order_data['OrderDate'].min(), order_data['OrderDate'].max())
