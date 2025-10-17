
import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv


from litellm import completion


st.set_page_config(page_title="LLM Investment Insight", layout="wide")
load_dotenv()  

DEFAULT_DATA = "data/merged_msft_with_signals.csv"  
MODEL = os.getenv("LLM_MODEL", "groq/llama3-70b-8192")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY is missing. Add it to your environment or Streamlit secrets to enable LLM insights.")


SYS_PROMPT = (
    "You are a professional financial analyst. "
    "Write concise, data-driven bullet points and a one-sentence conclusion. Avoid advice; provide analysis."
)

def row_context(row):
    return (
        f"Date: {row['date']:%Y-%m-%d}\n"
        f"Price PX: {row['PX']:.2f}\n"
        f"1d return: {row['ret_1d']:.3%}\n"
        f"RSI(14): {row['rsi_14']:.1f}\n"
        f"MACD: {row['macd']:.2f} | MACD signal: {row['macd_sig_9']:.2f}\n"
        f"SMA cross (1=short>long): {int(row['sma_cross'])}\n"
        f"Bollinger position: {row['bb_pos']:.2f}\n"
        f"Avg news sentiment: {row['avg_sentiment']:.2f}\n"
        f"Strategy signal (1=long, 0=flat): {int(row['signal'])}\n"
    )

def llm_insight_from_row(row, temperature=0.2, max_tokens=220):
    if not GROQ_API_KEY:
        return "LLM is disabled (no GROQ_API_KEY)."
    prompt = f"{SYS_PROMPT}\n\nContext:\n{row_context(row)}"
    resp = completion(
        model=MODEL,
        api_key=GROQ_API_KEY,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp["choices"][0]["message"]["content"].strip()


st.title("LLM-Powered Investment Insight Generator")

data_file = st.sidebar.text_input("Merged data CSV", DEFAULT_DATA)
df = pd.read_csv(data_file, parse_dates=["date"])


needed = {"date","PX","avg_sentiment","ret_1d","rsi_14","macd","macd_sig_9","bb_pos","sma_cross","signal"}
missing = needed - set(df.columns)
if missing:
    st.error(f"Missing columns in {data_file}: {sorted(missing)}")
    st.stop()

df = df.sort_values("date").reset_index(drop=True)


with st.expander("Price & Sentiment", expanded=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["PX"], name="Price", mode="lines"
    ))
    
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_sentiment"]*200, name="Sentiment (×200)",
        mode="lines", line=dict(dash="dash")
    ))
    
    long_idx = df.index[df["signal"] == 1]
    fig.add_trace(go.Scatter(
        x=df.loc[long_idx, "date"], y=df.loc[long_idx, "PX"],
        mode="markers", marker_symbol="triangle-up", marker_size=10,
        name="Long signal"
    ))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


with st.expander("Toy Backtest (strategy vs buy & hold)"):
    sig = df["signal"].fillna(0).shift(1).fillna(0)           # enter next day
    strat_ret = sig * df["ret_1d"].fillna(0)
    bh_ret = df["ret_1d"].fillna(0)
    strat_curve = (1 + strat_ret).cumprod()
    bh_curve = (1 + bh_ret).cumprod()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["date"], y=strat_curve, name="Strategy"))
    fig2.add_trace(go.Scatter(x=df["date"], y=bh_curve, name="Buy & Hold"))
    fig2.update_layout(yaxis_title="Growth of $1", height=380, margin=dict(l=20,r=20,t=30,b=10))
    st.plotly_chart(fig2, use_container_width=True)


col1, col2 = st.columns([2,1])
with col1:
    pick_date = st.date_input("Pick a date for an LLM insight", value=df["date"].iloc[-1].date(),
                              min_value=df["date"].min().date(), max_value=df["date"].max().date())
with col2:
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)

row = df.loc[df["date"].dt.date == pick_date]
if row.empty:
    st.info("No data for the selected date.")
else:
    row = row.iloc[-1]
    with st.spinner("Generating insight..."):
        text = llm_insight_from_row(row, temperature=temperature)
    st.subheader(f"LLM Insight — {pick_date}")
    st.write(text)

    
    buffer = io.StringIO()
    buffer.write(f"Date: {pick_date}\n\n")
    buffer.write(text + "\n")
    st.download_button("Download insight (.txt)",
                       data=buffer.getvalue(),
                       file_name=f"llm_insight_{pick_date}.txt",
                       mime="text/plain")
