#Question: How would you develop a Python program to calculate the frequency distribution of words  in a text document? 
from collections import Counter

file = open("sample_text.txt", "r")

text = file.read().lower()

file.close()

words = text.replace(".", "").replace(",", "").split()

frequency = Counter(words)

print("Word Frequency Distribution\n")

for word, count in frequency.items():
    print(word, ":", count)
