import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [1200, 1800, 1500, 2200, 2500, 2800]

plt.plot(months, sales, marker='o')
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)

plt.show()


import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [1200, 1800, 1500, 2200, 2500, 2800]

plt.scatter(months, sales)

plt.title("Monthly Sales Scatter Plot")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.show()



import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [1200, 1800, 1500, 2200, 2500, 2800]

plt.bar(months, sales)

plt.title("Monthly Sales Bar Chart")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.show()
