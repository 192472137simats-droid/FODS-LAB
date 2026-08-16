import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("soccer_players.csv")

print("Soccer Players Data")
print(df)

print("\nTop 5 Players by Goals Scored")
print(df.sort_values(by="Goals", ascending=False).head(5)[["Name", "Goals"]])

print("\nTop 5 Players by Weekly Salary")
print(df.sort_values(by="WeeklySalary", ascending=False).head(5)[["Name", "WeeklySalary"]])

average_age = df["Age"].mean()
print("\nAverage Age of Players =", round(average_age, 2))

print("\nPlayers Above the Average Age")
print(df[df["Age"] > average_age]["Name"].tolist())

# Distribution of players based on position
position_counts = df["Position"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(position_counts.index, position_counts.values)
plt.title("Distribution of Players by Position")
plt.xlabel("Position")
plt.ylabel("Number of Players")
plt.show()
