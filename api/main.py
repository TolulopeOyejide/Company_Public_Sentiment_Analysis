from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from src.schemas import PredictRequest, PredictResponse, AnalyzeTwitterResponse
from src.predict import predict
from src.ingest_twitter import fetch_tweets

app = FastAPI(title="Company Public Sentiment API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
    labels, probs = predict(req.texts)
    return PredictResponse(labels=labels, probabilities=probs)

@app.get("/analyze_twitter", response_model=AnalyzeTwitterResponse)
async def analyze_twitter(
    query: str = Query(..., description="Company name, brand, or keyword e.g. 'UBA Bank'"),
    max_results: int = Query(200, ge=10, le=1000)
):
    df = fetch_tweets(query=query, max_results=max_results)
    texts = df["text"].astype(str).tolist()
    labels, probs = predict(texts)

    results = [{"text": t, "label": l, "probability": float(p)} for t, l, p in zip(texts, labels, probs)]

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    examples = {"positive": [], "neutral": [], "negative": []}
    for r in results:
        lbl = r["label"].lower()
        if lbl not in counts:
            counts[lbl] = 0
            examples[lbl] = []
        counts[lbl] += 1
        if len(examples[lbl]) < 5:
            examples[lbl].append(r["text"])

    return AnalyzeTwitterResponse(
        query=query,
        total=len(results),
        counts=counts,
        examples=examples,
        results=results,
    )

# Run: uvicorn api.main:app --reload --port 8000
