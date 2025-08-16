import requests
import streamlit as st
from typing import List
import pandas as pd

st.set_page_config(page_title="Company Public Sentiment Analysis", page_icon="💬", layout="wide")

st.title("💬 Company Twitter/X Sentiment Analysis")
api_url = st.text_input("API URL", value="http://localhost:8000")

with st.sidebar:
    st.header("Analyze any company/brand")
    query = st.text_input("Company / Keyword", value="UBA Bank")
    max_results = st.slider("Max tweets", min_value=50, max_value=1000, value=300, step=50)
    fetch_go = st.button("Fetch & Analyze", type="primary")

st.write("Or test manually by pasting lines of text below (bypasses Twitter ingestion):")
raw = st.text_area("Manual Tweets (one per line)", height=160, placeholder="I love this product!\nThis is terrible.")
manual_go = st.button("Analyze Manual Input")

if manual_go:
    texts: List[str] = [line.strip() for line in raw.splitlines() if line.strip()]
    if not texts:
        st.warning("Please enter at least one line of text.")
    else:
        try:
            resp = requests.post(f"{api_url}/predict", json={"texts": texts}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame({
                "text": texts,
                "label": data["labels"],
                "probability": data["probabilities"],
            })
            st.subheader("Results")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df["label"].value_counts())
        except Exception as e:
            st.error(f"Failed to call API: {e}")

if fetch_go:
    if not query:
        st.warning("Please enter a company or keyword in the sidebar.")
    else:
        try:
            with st.spinner("Fetching tweets and analyzing…"):
                resp = requests.get(f"{api_url}/analyze_twitter", params={"query": query, "max_results": max_results}, timeout=120)
                resp.raise_for_status()
                data = resp.json()

            st.subheader(f"Sentiment for: {data['query']}")
            counts = data["counts"]
            counts_df = pd.DataFrame({"label": list(counts.keys()), "count": list(counts.values())})
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Distribution")
                st.bar_chart(counts_df.set_index("label"))
            with col2:
                st.caption("Examples")
                for lbl in ["positive", "neutral", "negative"]:
                    if lbl in data["examples"] and data["examples"][lbl]:
                        st.markdown(f"**{lbl.title()}**")
                        for t in data["examples"][lbl]:
                            st.write(f"• {t}")

            res_df = pd.DataFrame(data["results"])  # text,label,probability
            st.subheader("All analyzed tweets")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            csv = res_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"{data['query'].replace(' ','_')}_sentiment.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Failed to analyze: {e}")
