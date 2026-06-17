"""
run.py  —  One-shot launcher for SentimentScope
Usage:  python run.py
  • Installs all dependencies
  • Downloads NLTK data
  • Trains the model if model/sentiment_model.pkl is missing
  • Launches Streamlit
"""
import subprocess, sys, os

REQUIRED = [
    "streamlit==1.32.0",
    "scikit-learn==1.4.0",
    "pandas==2.2.0",
    "nltk==3.8.1",
    "joblib==1.3.2",
    "matplotlib==3.8.2",
    "seaborn==0.13.2",
    "numpy==1.26.4",
]

def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, **kw)

print("\n🔧  Installing dependencies…")
pkgs = " ".join(REQUIRED)
run(f'{sys.executable} -m pip install {pkgs} -q')

print("📦  Downloading NLTK data…")
run(f'{sys.executable} -c "import nltk; nltk.download(\'stopwords\',quiet=True); nltk.download(\'punkt\',quiet=True); nltk.download(\'punkt_tab\',quiet=True)"')

if not os.path.exists(os.path.join("model","sentiment_model.pkl")):
    print("🤖  Training model (first run only)…")
    run(f'{sys.executable} train_model.py')
else:
    print("✅  Model already trained — skipping.")

print("\n🚀  Launching SentimentScope at http://localhost:8501\n")
run("streamlit run app.py")
