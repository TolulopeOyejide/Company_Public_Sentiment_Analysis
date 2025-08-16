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
│   ├── predict.py        # For making predictions
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
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/twitter_sentiment_mlops.git
   cd twitter_sentiment_mlops
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Twitter API credentials to an `.env` file:
   ```env
   TWITTER_BEARER_TOKEN = Your bearer token
   PROJECT_NAME = company-public-sentiment-analysis
   ```

4. Run data ingestion:
   ```bash
   python src/ingest_twitter.py --company "Tesla"
   ```

5. Train the model:
   ```bash
  python -m src.train
   ```

6. Serve the model API:
   ```bash
   uvicorn api.main:app --reload
   ```



7. Build and run with Docker:
   ```bash
   docker build -t company-public-sentiment-analysis .
   docker run -p 8000:8000 company-public-sentiment-analysis
   ```

8. View the UI
`streamlit run app/streamlit_app.py`
