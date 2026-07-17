import matplotlib.pyplot as plt

months = ['Jan','Feb','Mar','Apr','May','Jun']
sales = [15000, 18000, 12000, 22000, 19000, 25000]

# 1. Line plot
plt.figure(figsize=(6,4))
plt.plot(months, sales, marker='o', color='blue')
plt.title('Monthly Sales - Line Plot')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# 2. Bar plot
plt.figure(figsize=(6,4))
plt.bar(months, sales, color='orange')
plt.title('Monthly Sales - Bar Plot')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()
