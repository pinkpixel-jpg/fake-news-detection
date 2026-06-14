import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

data = pd.read_csv("Dataset/cleaned_news.csv")

ps = PorterStemmer()

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

data["processed_text"] = data["text"].apply(preprocess)

data.to_csv("Dataset/processed_news.csv", index=False)

print("Preprocessing completed successfully!")
print(data.columns)