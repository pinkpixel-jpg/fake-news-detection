import math
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("models/bert")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset(csv_path: str = "Dataset/cleaned_news.csv") -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)
    dataset = dataset.dropna(subset=["title", "text", "label"]).copy()
    dataset["combined_text"] = (
        dataset["title"].fillna("") + ". " + dataset["text"].fillna("")
    )
    dataset["label"] = dataset["label"].astype(int)
    return dataset[["combined_text", "label"]]


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    correct = sum(p == y for p, y in zip(all_predictions, all_labels))
    return correct / len(all_labels)


def train_bert(epochs: int = 2, batch_size: int = 8, learning_rate: float = 2e-5):
    dataset = build_dataset()
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        dataset["combined_text"].tolist(),
        dataset["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"].tolist(),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    eval_dataset = NewsDataset(eval_texts, eval_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, math.floor(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | "
                    f"Loss: {running_loss / step:.4f}"
                )

        eval_accuracy = evaluate(model, eval_loader, device)
        print(f"Epoch {epoch} completed. Eval accuracy: {eval_accuracy:.4f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned BERT model to {OUTPUT_DIR}")


if __name__ == "__main__":
    train_bert()
