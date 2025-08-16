import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import mlflow

from .config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, MLFLOW_TRACKING_URI
from .utils import ensure_dirs
from .preprocess import normalize_batch

def load_training_data() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "labeled_tweets.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Provide a labeled dataset at data/raw/labeled_tweets.csv with columns text,label"
        )
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("labeled_tweets.csv must have columns: text,label")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=None, tokenizer=None)),
        ("clf", MultinomialNB()),
    ])

def main():
    ensure_dirs(PROCESSED_DIR, MODELS_DIR)

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    mlflow.set_experiment("twitter_sentiment_nb")

    df = load_training_data()
    df["text_norm"] = normalize_batch(df["text"].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_norm"], df["label"], test_size=2, random_state=42)

    pipe = build_pipeline()

    with mlflow.start_run():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        report = classification_report(y_test, preds, output_dict=True)
        for k, v in report["weighted avg"].items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"weighted_{k}", v)

        model_path = os.path.join(MODELS_DIR, "model.joblib")
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        print("Model saved to", model_path)

if __name__ == "__main__":
    main()
