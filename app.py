import pickle
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Code.utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
BERT_MODEL_DIR = MODELS_DIR / "bert"

app = FastAPI(
    title="Fake News Detection",
    description="A professional website for fake news classification using a baseline model and BERT.",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

baseline_model = None
vectorizer = None
bert_tokenizer = None
bert_model = None
baseline_ready = False
bert_ready = False


def load_baseline():
    global baseline_model, vectorizer, baseline_ready
    try:
        with open(BASELINE_MODEL_PATH, "rb") as f:
            baseline_model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        baseline_ready = True
    except FileNotFoundError:
        baseline_ready = False


def load_bert():
    global bert_tokenizer, bert_model, bert_ready
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
        bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        bert_model.eval()
        bert_ready = True
    except Exception:
        bert_ready = False


def predict_baseline_text(text: str) -> Optional[str]:
    if not baseline_ready or baseline_model is None or vectorizer is None:
        return None
    cleaned = clean_text(text)
    transformed = vectorizer.transform([cleaned])
    prediction = baseline_model.predict(transformed)[0]
    return "Real" if prediction == 1 else "Fake"


def predict_bert_text(text: str) -> Optional[dict]:
    if not bert_ready or bert_tokenizer is None or bert_model is None:
        return None
    inputs = bert_tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0].tolist()
        label = "Real" if scores[1] > scores[0] else "Fake"
        return {"label": label, "scores": scores}


@app.on_event("startup")
def startup_event():
    load_baseline()
    load_bert()


@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "baseline_ready": baseline_ready,
            "bert_ready": bert_ready,
            "result": None,
            "news_text": "",
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, news_text: str = Form(...)):
    baseline_result = predict_baseline_text(news_text)
    bert_result = predict_bert_text(news_text)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "baseline_ready": baseline_ready,
            "bert_ready": bert_ready,
            "news_text": news_text,
            "baseline_result": baseline_result,
            "bert_result": bert_result,
        },
    )
