import pandas as pd

# Load merged dataset
data = pd.read_csv("Dataset/merged_news.csv")

print("Original Shape:", data.shape)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

print("\nShape After Removing Null Values:", data.shape)

# Save cleaned dataset
data.to_csv("Dataset/cleaned_news.csv", index=False)

print("\nCleaned dataset saved successfully!")