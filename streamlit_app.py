# streamlit_app.py
import os, math, json, requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# -------------------- Page setup & style --------------------
st.set_page_config(page_title="Memecoin Dashboard", layout="wide")
st.markdown('''
<style>
.block-container {padding-top: 2rem;}
h1 {background: linear-gradient(90deg, #7C4DFF, #4DD0E1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
</style>
''', unsafe_allow_html=True)

st.title("🚀 Memecoin Dashboard — Beginner Edition")
st.caption("Discover • Score • Track — with safe defaults and visuals")

load_dotenv()

# -------------------- Controls --------------------
mode = st.radio("Data source mode", ["Demo (offline)"], horizontal=True)
chains = st.multiselect("Chains", ["Ethereum", "Solana", "BSC"],
                        default=["Ethereum", "Solana", "BSC"])

colA, colB, colC, colD = st.columns(4)
with colA:
    min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=200_000, step=10_000)
with colB:
    min_vol = st.number_input("Min 24h Volume ($)", min_value=0, value=75_000, step=5_000)
with colC:
    max_age = st.number_input("Max Age (days)", min_value=0, value=180, step=1)
with colD:
    require_locked = st.checkbox("Require Liquidity Locked ≥ 70%", value=True)

st.subheader("Weights (normalized)")
defaults = {
    "w_liq": 20, "w_vol": 20, "w_tx": 10, "w_age": 5,
    "w_lock": 20, "w_top10": 20, "w_sent": 15, "w_security": 10
}
weights = {}
labels = [
    ("w_liq", "Liquidity"),
    ("w_vol", "24h Volume"),
    ("w_tx", "Txns"),
    ("w_age", "Age (younger=better)"),
    ("w_lock", "Liquidity Locked %"),
    ("w_top10", "Top-10 Holders % (lower better)"),
    ("w_sent", "Sentiment"),
    ("w_security", "Security"),
]
for i, (k, lab) in enumerate(labels):
    with st.columns(4)[i % 4]:
        weights[k] = st.slider(lab, 0, 100, defaults[k], 1)
wtot = sum(weights.values()) or 1
for k in weights:
    weights[k] = weights[k] / wtot

# -------------------- Load demo data --------------------
@st.cache_data
def load_demo():
    # Expects sample_data.csv in repo root
    return pd.read_csv("sample_data.csv")

try:
    data = load_demo()
except FileNotFoundError:
    st.error("sample_data.csv not found. Add it to the same folder as streamlit_app.py.")
    st.stop()

# Ensure required columns exist (in case you swap datasets later)
required_cols = [
    "name","symbol","chain","price","liquidity_usd","volume24h_usd","txns24h",
    "mcap_usd","age_days","is_honeypot","owner_renounced","liquidity_locked_pct",
    "top10_holders_pct","telegram_members","twitter_followers","sentiment_score"
]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"The dataset is missing columns: {missing}")
    st.stop()

df = data.copy()

# -------------------- Filtering --------------------
def apply_filters(df_in):
    df1 = df_in[df_in["chain"].isin(chains)]
    df1 = df1[df1["liquidity_usd"] >= min_liq]
    df1 = df1[df1["volume24h_usd"] >= min_vol]
    df1 = df1[df1["age_days"] <= max_age]
    if require_locked and "liquidity_locked_pct" in df1.columns:
        df1 = df1[df1["liquidity_locked_pct"].fillna(0) >= 70]
    hide_honeypots = st.checkbox("Hide suspected honeypots", value=True)
    if "is_honeypot" in df1.columns and hide_honeypots:
        df1 = df1[~df1["is_honeypot"].fillna(False)]
    return df1

filtered = apply_filters(df)

# If filters eliminate everything, auto-relax so the user still sees results
if filtered.empty:
    st.warning("No projects match your filters. I’ll relax thresholds so you can see results. "
               "Try lowering min liquidity/volume or turning off the lock requirement.")
    # Relax strategy: drop lock requirement, lower thresholds to 25% of input
    relaxed = df.copy()
    relaxed = relaxed[relaxed["chain"].isin(chains)]
    relaxed = relaxed[relaxed["liquidity_usd"] >= max(0, min_liq * 0.25)]
    relaxed = relaxed[relaxed["volume24h_usd"] >= max(0, min_vol * 0.25)]
    relaxed = relaxed[relaxed["age_days"] <= max_age if max_age > 0 else relaxed["age_days"].max()]
    filtered = relaxed  # show relaxed results

if filtered.empty:
    st.error("Still no data to show after relaxing. Please broaden your filters.")
    st.stop()

# -------------------- Scoring --------------------
def minmax(s, invert=False):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if s.max() == s.min():
        norm = pd.Series(0.5, index=s.index)
    else:
        norm = (s - s.min()) / (s.max() - s.min())
    return 1 - norm if invert else norm

scored = filtered.copy()
scored["score"] = (
    weights["w_liq"]  * minmax(scored["liquidity_usd"]) +
    weights["w_vol"]  * minmax(scored["volume24h_usd"]) +
    weights["w_tx"]   * minmax(scored["txns24h"]) +
    weights["w_age"]  * (1 - minmax(scored["age_days"])) +
    weights["w_lock"] * minmax(scored["liquidity_locked_pct"]) +
    weights["w_top10"]* (1 - minmax(scored["top10_holders_pct"])) +
    weights["w_sent"] * minmax(scored["sentiment_score"]) +
    weights["w_security"] * (1 - minmax(scored["is_honeypot"].astype(int)))
).round(3)

ranked = scored.sort_values("score", ascending=False).reset_index(drop=True)

# -------------------- Table --------------------
st.markdown("### Ranked Results")
st.dataframe(
    ranked[[
        "score","name","symbol","chain","price","liquidity_usd","volume24h_usd","txns24h",
        "mcap_usd","age_days","liquidity_locked_pct","top10_holders_pct","sentiment_score"
    ]],
    use_container_width=True
)

# -------------------- Visual Overview --------------------
st.markdown("### 📊 Visual Overview")
tab1, tab2, tab3 = st.tabs(["Overview", "Details", "Notes"])

with tab1:
    # Metric cards for top 3
    top3 = ranked.head(3)
    cols = st.columns(3)
    for i, (_, r) in enumerate(top3.iterrows()):
        with cols[i]:
            st.subheader(f"{r['name']} ({r['symbol']})")
            st.metric("Score", f"{r['score']:.3f}")
            st.metric("Liquidity", f"${r['liquidity_usd']:,.0f}")
            st.metric("24h Volume", f"${r['volume24h_usd']:,.0f}")
            st.metric("Top-10 Holders %", f"{float(r.get('top10_holders_pct', 0)):.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        top_vol = ranked.nlargest(10, "volume24h_usd")[["name","volume24h_usd","score"]]
        fig_bar = px.bar(top_vol, x="name", y="volume24h_usd",
                         hover_data=["score"], title="Top 10 by 24h Volume")
        fig_bar.update_layout(xaxis_title="", yaxis_title="24h Volume (USD)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        fig_scatter = px.scatter(
            ranked.head(100),
            x="liquidity_usd", y="volume24h_usd",
            size="mcap_usd", color="score",
            hover_name="name", title="Liquidity vs Volume"
        )
        fig_scatter.update_layout(xaxis_title="Liquidity (USD)", yaxis_title="24h Volume (USD)")
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    # Safe selection logic
    options = ranked["name"].tolist()
    if not options:
        st.warning("No projects available after filters. Adjust filters above.")
        st.stop()
    sel = st.selectbox("Pick a project", options)
    if sel not in options:
        sel = options[0]
    row = ranked.loc[ranked["name"] == sel]
    if row.empty:
        row = ranked.iloc[[0]]
        sel = row.iloc[0]["name"]
    row = row.iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", row["score"])
        st.metric("Price", f"${float(row['price']):.10f}")
    with col2:
        st.metric("Liquidity", f"${float(row['liquidity_usd']):,.0f}")
        st.metric("Mcap", f"${float(row['mcap_usd']):,.0f}")
    with col3:
        st.metric("Age", int(row['age_days']))
        st.metric("Txns24h", int(row['txns24h']))

    # Radar chart
    def _norm(col, invert=False):
        return float(minmax(ranked[col], invert).loc[ranked["name"] == sel])

    radar_vals = {
        "Liquidity": _norm("liquidity_usd"),
        "Volume": _norm("volume24h_usd"),
        "Txns": _norm("txns24h"),
        "Lock%": _norm("liquidity_locked_pct"),
        "Top10(↓)": _norm("top10_holders_pct", invert=True),
        "Sentiment": _norm("sentiment_score"),
    }
    theta = list(radar_vals.keys())
    rvals = list(radar_vals.values()) + [list(radar_vals.values())[0]]
    fig = go.Figure(data=[go.Scatterpolar(r=rvals, theta=theta + [theta[0]],
                                          fill='toself', name=sel)])
    fig.update_layout(title="Attribute Radar",
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("Use this tab as your trading journal — jot down entry/exit reasoning, catalysts, risks, etc.")
    st.download_button(
        "Export current table (CSV)",
        data=ranked.to_csv(index=False).encode("utf-8"),
        file_name="memecoin_ranked_export.csv",
        mime="text/csv"
    )
