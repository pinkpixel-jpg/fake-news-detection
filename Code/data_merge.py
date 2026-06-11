import pandas as pd
import os

print("Current working directory:", os.getcwd())
import os

print("Current working directory:", os.getcwd())
print("Dataset exists:", os.path.exists("Dataset/Fake.csv"))
fake = pd.read_csv("Dataset/Fake.csv")
true = pd.read_csv("Dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

data.to_csv("Dataset/merged_news.csv", index=False)

print("Dataset merged successfully!")
print("Shape:", data.shape)