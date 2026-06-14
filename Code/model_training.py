import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load processed dataset
data = pd.read_csv("Dataset/processed_news.csv")

print("Original Shape:", data.shape)

# Remove rows where processed_text is missing
data = data.dropna(subset=["processed_text"])

print("Shape After Removing Null Values:", data.shape)

# Features and Labels
X = data["processed_text"].fillna("")
y = data["label"]

# Convert text into numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

print("TF-IDF Transformation Completed")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# Save Model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved as model.pkl")
print("Vectorizer Saved as vectorizer.pkl")