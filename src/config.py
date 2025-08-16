import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "company-public-sentiment-analysis")
DATA_DIR = os.path.join(os.getcwd(), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(os.getcwd(), "models", "latest")

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "") or os.path.join(os.getcwd(), "mlruns")
