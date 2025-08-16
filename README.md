# Company_Public_Sentiment_Analysis
Generic end-to-end MLOps pipeline for performing sentiment analysis on Twitter(now X) data for any company or brand.

This repository contains an end-to-end MLOps pipeline for performing sentiment analysis on tweets for any company.
You can easily adapt it by providing a company name or keyword, and the pipeline will fetch tweets, clean the data,
train a sentiment analysis model, log experiments with MLflow, and serve the model via a REST API.

## Features
- Fetch tweets for any company using the Twitter API (Tweepy)
- Data cleaning and preprocessing
- Sentiment labeling using VADER SentimentIntensityAnalyzer
- Model training with Logistic Regression
- Experiment tracking with MLflow
- Model serving using FastAPI
- Dockerized deployment

## Project Structure
```
twitter_sentiment_mlops/
│
├── data/                   # Raw and processed datasets
├── src/                    # Source code for the pipeline
│   ├── ingest_twitter.py   # Fetches tweets
│   ├── preprocess.py       # Cleans and preprocesses text
│   ├── train.py            # Trains and logs the model
│   ├── predict.py          # For making predictions
│   └── utils.py            # Helper functions
├── models/                 # Saved models
├── Dockerfile              # Docker setup
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Requirements
- Python 3.8+
- Twitter Developer account & API keys
- MLflow
- FastAPI
- Docker

## Setup Instructions
1. Clone the repository:  <br>
   `git clone https://github.com/TolulopeOyejide/Company_Public_Sentiment_Analysis.git`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Twitter API credentials to an `.env` file:
   ```env
   TWITTER_BEARER_TOKEN = Your bearer token
   PROJECT_NAME = company-public-sentiment-analysis
   ```

4. Train the model: <br>
   `python -m src.train`


5. Serve the model API:  <br>
  `uvicorn api.main:app --reload` <br>

6. Build and run with Docker:  <br>
   `docker build -t company-public-sentiment-analysis .` <br>
   `docker run -p 8000:8000 company-public-sentiment-analysis`  <br>
   
7. View the UI  <br>
   `streamlit run app/streamlit_app.py`
