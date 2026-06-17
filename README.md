# 🧠 SentimentScope

> Sentiment Analysis Web App — 100% free, runs fully offline, no API keys needed.

---

## ⚡ Quickstart (3 ways)

### Option 1 — One command (recommended)
```bash
python run.py
```
This installs everything, trains the model, and opens the app automatically.

---

### Option 2 — Manual steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (only needed once)
python train_model.py

# 3. Launch the app
streamlit run app.py
```
Open → **http://localhost:8501**

---

### Option 3 — Virtual environment (cleanest)
```bash
# Create venv
python -m venv venv

# Activate  (Windows)
venv\Scripts\activate

# Activate  (Mac / Linux)
source venv/bin/activate

# Install + run
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

---

## 📂 Project Structure

```
sentiment-analysis/
├── run.py              ← One-shot launcher (install + train + run)
├── app.py              ← Streamlit web app
├── train_model.py      ← ML training pipeline
├── utils.py            ← Text cleaning utilities
├── requirements.txt    ← All dependencies (pinned)
├── model/
│   ├── sentiment_model.pkl     ← Trained model (pre-built ✅)
│   ├── vectorizer.pkl          ← TF-IDF vectoriser (pre-built ✅)
│   └── confusion_matrix.png    ← Evaluation chart
└── data/
    └── dataset.csv             ← Training dataset (8,000 rows included)
```

> The `model/` folder is pre-built. You can run `streamlit run app.py` directly without training.

---

## ✨ Features

| Feature | Details |
|---|---|
| Single text analysis | Paste any review, tweet, or comment |
| Confidence score | Visual bar showing model certainty |
| Class probabilities | Breakdown across Positive / Neutral / Negative |
| Batch CSV upload | Analyse thousands of rows at once |
| Download results | Export batch predictions as CSV |
| Sentiment chart | Distribution bar chart for batch results |
| Confusion matrix | Model evaluation on the test set |
| Dark UI | High-contrast, fully readable theme |

---

## 🗂 Using a Real Dataset (optional)

Replace `data/dataset.csv` with a real dataset for higher accuracy:

| Dataset | URL | Accuracy |
|---|---|---|
| IMDb 50k | kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews | ~89–91% |
| Twitter Sentiment140 | kaggle.com/datasets/kazanova/sentiment140 | ~85–88% |

Then re-run `python train_model.py`.

---

## 🧠 ML Pipeline

```
Raw Text → Clean → TF-IDF (50k features, bigrams) → Logistic Regression → Label + Probability
```

- **Cleaning**: lowercase, remove URLs/HTML/punctuation/stopwords (negations kept)
- **Vectoriser**: TF-IDF, max 50k features, unigram+bigram, sublinear TF
- **Model**: Logistic Regression, C=1.0, lbfgs, 500 iterations

---

## 🌐 Deploy Free on Streamlit Cloud

```bash
git init && git add . && git commit -m "SentimentScope"
git remote add origin https://github.com/YOUR_USERNAME/sentimentscope.git
git push -u origin main
```
Then go to → **https://share.streamlit.io** → New App → Deploy.

---
