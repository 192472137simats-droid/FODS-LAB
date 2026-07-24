import pandas as pd

sales_data = pd.DataFrame({
    'Product': ['A','B','C','D','E','F','G'],
    'Quantity_Sold': [120, 95, 200, 60, 150, 300, 80]
})

top5 = sales_data.sort_values('Quantity_Sold', ascending=False).head(5)
print(top5)
