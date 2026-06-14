import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    stemmed = [ps.stem(word) for word in filtered]
    return " ".join(stemmed)
