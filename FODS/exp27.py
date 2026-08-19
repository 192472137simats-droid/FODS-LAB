#Scenario: You are a data analyst working for a sports analytics company. The company has collected data on various soccer players, including their names, ages, positions, number of goals scored, and weekly salaries. Create dataset on your own and store in a CSV file. Question: Develop a Python program to read the data from the CSV file into a pandas data frame, to find the top 5 players with the highest number of goals scored and the top 5 players with the highest salaries. Also calculate the average age of players and display the names of players who are above the average age and visualize the distribution of players based on their positions using a bar chart. 
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
