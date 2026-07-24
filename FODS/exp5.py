import numpy as np

# Fuel efficiency (mpg) of different car models
fuel_efficiency = np.array([25, 30, 28, 35, 40])

# Average fuel efficiency
average_efficiency = np.mean(fuel_efficiency)

# Percentage improvement between two car models
model_a = fuel_efficiency[0]   # e.g., Car 1
model_b = fuel_efficiency[3]   # e.g., Car 4

percentage_improvement = ((model_b - model_a) / model_a) * 100

print("Fuel Efficiency (mpg):", fuel_efficiency)
print("Average fuel efficiency:", average_efficiency)
print(f"Percentage improvement (Model A -> Model B): {percentage_improvement:.2f}%")
