import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Load processed dataset
data = pd.read_csv("Dataset/processed_news.csv")

print("Original Shape:", data.shape)

# Remove null values
data = data.dropna(subset=["processed_text"])

# Features and Labels
X = data["processed_text"].fillna("")
y = data["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION RESULTS")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))