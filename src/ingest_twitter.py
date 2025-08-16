from __future__ import annotations
import os
import pandas as pd
from typing import List
from .config import RAW_DIR, TWITTER_BEARER_TOKEN
from .utils import ensure_dirs

def _fetch_with_tweepy(query: str, max_results: int) -> List[str]:
    import tweepy
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
    texts: List[str] = []
    for tweet in tweepy.Paginator(
        client.search_recent_tweets,
        query=f"{query} -is:retweet lang:en",
        tweet_fields=["id", "text", "lang"],
        max_results=100,
    ).flatten(limit=max_results):
        if getattr(tweet, "lang", "en") == "en":
            texts.append(tweet.text)
    return texts

def _fetch_with_snscrape(query: str, max_results: int) -> List[str]:
    import subprocess, json
    cmd = [
        "snscrape", "--jsonl", "--max-results", str(max_results), "twitter-search",
        f"{query} lang:en"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    texts: List[str] = []
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
            texts.append(obj.get("content", ""))
        except Exception:
            continue
    return texts

def fetch_tweets(query: str, max_results: int = 200) -> pd.DataFrame:
    ensure_dirs(RAW_DIR)
    texts: List[str] = []
    if TWITTER_BEARER_TOKEN:
        try:
            texts = _fetch_with_tweepy(query, max_results)
        except Exception:
            texts = []
    if not texts:
        try:
            texts = _fetch_with_snscrape(query, max_results)
        except Exception:
            texts = []
    if not texts:
        csv_path = os.path.join(RAW_DIR, "tweets.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "text" not in df.columns:
                raise ValueError("tweets.csv must contain a 'text' column")
            return df[["text"]].dropna().reset_index(drop=True)
        raise RuntimeError("No Twitter data source available. Set TWITTER_BEARER_TOKEN, install snscrape, or place data/raw/tweets.csv")

    return pd.DataFrame({"text": texts})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="(data OR ai OR ml)")
    parser.add_argument("--max_results", type=int, default=200)
    parser.add_argument("--out", type=str, default=os.path.join(RAW_DIR, "tweets.csv"))
    args = parser.parse_args()

    df = fetch_tweets(args.query, args.max_results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} tweets -> {args.out}")
