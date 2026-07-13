# Item prices and quantities
item_prices = [50, 120, 30, 200]
quantities = [2, 1, 5, 1]

discount_rate = 10   # in %
tax_rate = 8          # in %

# Subtotal = sum of (price * quantity) for each item
subtotal = sum(price * qty for price, qty in zip(item_prices, quantities))

# Apply discount
discount_amount = (discount_rate / 100) * subtotal
amount_after_discount = subtotal - discount_amount

# Apply tax
tax_amount = (tax_rate / 100) * amount_after_discount
total_cost = amount_after_discount + tax_amount

print("Subtotal:", subtotal)
print("Discount:", discount_amount)
print("Amount after discount:", amount_after_discount)
print("Tax:", tax_amount)
print("Total cost:", round(total_cost, 2))
