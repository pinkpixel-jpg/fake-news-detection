import pandas as pd

fake = pd.read_csv("../Dataset/Fake.csv")
true = pd.read_csv("../Dataset/True.csv")

print("Fake Dataset Shape:", fake.shape)
print("True Dataset Shape:", true.shape)

print("\nFake Dataset Columns:")
print(fake.columns)

print("\nTrue Dataset Columns:")
print(true.columns)

print("\nFirst 5 Rows of Fake Dataset:")
print(fake.head())

print("\nFirst 5 Rows of True Dataset:")
print(true.head())