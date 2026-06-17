# train_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Complete training pipeline:
#   1. Load dataset
#   2. Map labels to 0 / 1 / 2  (Negative / Neutral / Positive)
#   3. Clean text with utils.clean_text
#   4. TF-IDF vectorisation
#   5. Train Logistic Regression
#   6. Evaluate: accuracy + confusion matrix
#   7. Save model and vectoriser with joblib
#
# Run with:  python train_model.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (
    accuracy_score, classification_report, confusion_matrix
)

from utils import clean_text   # our shared cleaning function

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH       = os.path.join("data", "dataset.csv")
MODEL_DIR       = "model"
MODEL_PATH      = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CM_IMAGE_PATH   = os.path.join(MODEL_DIR, "confusion_matrix.png")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load Dataset ───────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the CSV and normalises column names to 'text' and 'label'.

    Supported dataset formats
    ─────────────────────────
    A) Twitter Sentiment140  (kaggle.com/datasets/kazanova/sentiment140)
       Columns: target, ids, date, flag, user, text
       target: 0 = negative, 2 = neutral (not in dataset), 4 = positive

    B) IMDb Movie Reviews    (kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
       Columns: review, sentiment   (sentiment: 'positive' / 'negative')

    C) Any custom CSV with columns 'text' and 'sentiment' or 'label'.
    """
    if not os.path.exists(path):
        print(f"\n❌  Dataset not found at '{path}'")
        print(    "    Please follow README.md → Dataset section to download it.\n")
        sys.exit(1)

    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"✅  Loaded {len(df):,} rows.  Columns: {list(df.columns)}")

    # ── Detect and normalise format ──────────────────────────────────────────
    col_lower = {c.lower(): c for c in df.columns}

    # Format A: Sentiment140
    if "target" in col_lower and "text" in col_lower:
        df = df[[col_lower["target"], col_lower["text"]]].copy()
        df.columns = ["label", "text"]
        # Sentiment140: 0 → negative (0), 4 → positive (2); no neutral
        df["label"] = df["label"].map({0: 0, 4: 2})
        df.dropna(subset=["label"], inplace=True)

    # Format B: IMDb
    elif "sentiment" in col_lower and ("review" in col_lower or "text" in col_lower):
        text_col = "review" if "review" in col_lower else "text"
        df = df[[col_lower[text_col], col_lower["sentiment"]]].copy()
        df.columns = ["text", "label"]
        df["label"] = df["label"].str.lower().map(
            {"positive": 2, "negative": 0, "neutral": 1}
        )
        df.dropna(subset=["label"], inplace=True)

    # Format C: generic (columns must be 'text' + 'label' or 'sentiment')
    elif "text" in col_lower:
        lbl_col = "label" if "label" in col_lower else "sentiment"
        df = df[[col_lower["text"], col_lower[lbl_col]]].copy()
        df.columns = ["text", "label"]
        # If label is string, try to map common values
        if df["label"].dtype == object:
            mapping = {"positive": 2, "negative": 0, "neutral": 1,
                       "pos": 2, "neg": 0, "neu": 1, "1": 2, "0": 0, "-1": 0}
            df["label"] = df["label"].str.lower().map(mapping)
        df.dropna(subset=["label"], inplace=True)

    else:
        print("❌  Could not detect dataset format. See README.md for instructions.")
        sys.exit(1)

    df["label"] = df["label"].astype(int)
    df.dropna(subset=["text"], inplace=True)
    df = df[df["text"].str.strip() != ""]

    print(f"   Label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


# ── 2. Clean Text ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to every row. Shows a progress update."""
    print("🔄  Cleaning text (this may take a minute for large datasets)…")
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    # Drop rows that became empty after cleaning
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    print(f"   {len(df):,} usable rows after cleaning.\n")
    return df


# ── 3. TF-IDF + Train/Test Split ─────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    """
    Vectorise with TF-IDF and split into train / test sets.

    TF-IDF parameters:
      max_features=50000  — vocabulary size cap (keeps memory reasonable)
      ngram_range=(1,2)   — unigrams + bigrams (captures "not good", "very bad")
      sublinear_tf=True   — log-scale TF to reduce dominance of frequent terms
      min_df=2            — ignore terms that appear in fewer than 2 documents
    """
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"📊  Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


# ── 4. Train Model ────────────────────────────────────────────────────────────
def train(X_train_vec, y_train) -> LogisticRegression:
    """
    Logistic Regression with:
      C=1.0         — default regularisation
      max_iter=500  — enough for convergence on text data
      solver=lbfgs  — efficient multi-class solver
    """
    print("🤖  Training Logistic Regression…")
    model = LogisticRegression(
        C=1.0,
        max_iter=500,
        solver="lbfgs",
    )
    model.fit(X_train_vec, y_train)
    print("   Training complete.\n")
    return model


# ── 5. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model, X_test_vec, y_test):
    """Print accuracy + full classification report. Save confusion matrix PNG."""
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅  Test Accuracy : {acc*100:.2f}%\n")
    print("📋  Classification Report:")
    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    unique_labels = sorted(y_test.unique())
    target_names  = [label_names[l] for l in unique_labels]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_IMAGE_PATH, dpi=120)
    plt.close()
    print(f"   Confusion matrix saved → {CM_IMAGE_PATH}\n")


# ── 6. Save artifacts ─────────────────────────────────────────────────────────
def save_artifacts(model, vectorizer):
    joblib.dump(model,      MODEL_PATH,      compress=3)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
    print(f"💾  Model saved      → {MODEL_PATH}")
    print(f"💾  Vectorizer saved → {VECTORIZER_PATH}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Sentiment Analysis — Training Pipeline")
    print("=" * 60 + "\n")

    df              = load_dataset(DATA_PATH)
    df              = preprocess(df)
    X_tr, X_te, y_tr, y_te, vec = build_features(df)
    model           = train(X_tr, y_tr)
    evaluate(model, X_te, y_te)
    save_artifacts(model, vec)

    print("🎉  Done! Run the app with:  streamlit run app.py")
