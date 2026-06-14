# Fake News Detection using NLP

This project aims to classify news articles as Fake or Real using Natural Language Processing and Machine Learning techniques.

## Objectives
- Detect fake news
- Perform NLP preprocessing
- Train classification models
- Evaluate performance

## Tools
- Python
- Pandas
- NLTK
- Scikit-learn

## Usage

1. Create and activate the Python virtual environment (Windows PowerShell):

	```powershell
	python -m venv venv
	& .\venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	```

2. Prepare data (preprocessing):

	```powershell
	python Code/preprocessing.py
	```

3. Train the baseline model (TF-IDF + Logistic Regression):

	```powershell
	python Code/model_training.py
	```

4. (Optional, long) Train BERT fine-tuning:

	```powershell
	python Code/train_bert.py
	```

5. Run the web UI (default on port 8002):

	```powershell
	python -m uvicorn app:app --host 127.0.0.1 --port 8002
	```

6. Open `http://127.0.0.1:8002/` in your browser and paste news text to get predictions.

Notes:
- BERT training benefits greatly from a GPU. For CPU-only machines consider lowering dataset size or model size.
