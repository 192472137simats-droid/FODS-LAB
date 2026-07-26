import pandas as pd

data = {
    "Post_ID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Likes": [120, 250, 120, 300, 250, 180, 120, 300]
}

df = pd.DataFrame(data)

print("Post Data")
print(df)

frequency = df["Likes"].value_counts().sort_index()

print("\nFrequency Distribution of Likes")
print(frequency)
