import pandas as pd
import nltk

from Code.utils import clean_text

nltk.download("stopwords", quiet=True)

data = pd.read_csv("Dataset/cleaned_news.csv")

for col in ["title", "text"]:
    data[col] = data[col].fillna("")

combined = data["title"] + ". " + data["text"]
data["processed_text"] = combined.apply(clean_text)

data.to_csv("Dataset/processed_news.csv", index=False)

print("Preprocessing completed successfully!")
print(data.columns)