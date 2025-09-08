# streamlit_app.py
import os, time, json, requests
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
st.caption("Discover • Score • Track — Demo & Live modes with visuals")

load_dotenv()  # local .env; on Streamlit Cloud use Secrets

# -------------------- Controls --------------------
mode = st.radio("Data source mode", ["Demo (offline)", "Live (internet)"], horizontal=True)
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

# -------------------- Helpers --------------------
def minmax(s, invert=False):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if s.max() == s.min():
        norm = pd.Series(0.5, index=s.index)
    else:
        norm = (s - s.min()) / (s.max() - s.min())
    return 1 - norm if invert else norm

def normalize_chain(cid: str) -> str:
    cid = (cid or "").lower()
    if cid in ("eth","ethereum"): return "Ethereum"
    if cid in ("sol","solana"):   return "Solana"
    if cid in ("bsc","bnb"):      return "BSC"
    return cid.capitalize() if cid else "Unknown"

def to_float(x):
    try:
        if isinstance(x, dict):
            if "usd" in x: return float(x.get("usd") or 0)
            for v in x.values():
                try: return float(v)
                except: pass
            return 0.0
        if x is None: return 0.0
        return float(x)
    except:
        return 0.0

def to_int_txns_h24(tx):
    if tx is None: return 0
    # Expected shapes:
    # - number
    # - {"h24": 123}
    # - {"h24": {"buys": 100, "sells": 80}}
    if isinstance(tx, dict):
        h24 = tx.get("h24", 0)
        if isinstance(h24, dict):
            total = 0
            for v in h24.values():
                try: total += int(v or 0)
                except: pass
            return total
        try: return int(h24 or 0)
        except: return 0
    try:
        return int(tx or 0)
    except:
        return 0

def safe_get(url, headers=None, params=None, timeout=15):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, str(e)

def std_cols():
    return pd.DataFrame(columns=[
        "name","symbol","chain","price","liquidity_usd","volume24h_usd","txns24h",
        "mcap_usd","age_days","is_honeypot","owner_renounced","liquidity_locked_pct",
        "top10_holders_pct","telegram_members","twitter_followers","sentiment_score"
    ])

def rows_from_pairs(pairs, chains_selected):
    rows = []
    for p in pairs:
        chain = normalize_chain(p.get("chainId", ""))
        if chains_selected and chain not in chains_selected:
            continue
        liq_usd = to_float((p.get("liquidity") or {}).get("usd", 0))
        vol24   = to_float((p.get("volume") or {}).get("h24", 0))
        tx24    = to_int_txns_h24(p.get("txns"))
        fdv     = to_float(p.get("fdv"))
        price   = to_float(p.get("priceUsd"))
        base    = p.get("baseToken") or {}
        name    = base.get("name") or base.get("symbol") or "Unknown"
        sym     = base.get("symbol") or "?"
        rows.append(dict(
            name=name, symbol=sym, chain=chain,
            price=price, liquidity_usd=liq_usd, volume24h_usd=vol24, txns24h=tx24,
            mcap_usd=fdv, age_days=np.nan, is_honeypot=False, owner_renounced=False,
            liquidity_locked_pct=np.nan, top10_holders_pct=np.nan,
            telegram_members=np.nan, twitter_followers=np.nan, sentiment_score=np.nan
        ))
    return rows

# -------------------- Data loaders --------------------
@st.cache_data
def load_demo():
    return pd.read_csv("sample_data.csv")

@st.cache_data(ttl=180)
def load_live(chains_selected):
    meta = {"source": "dexscreener_trending", "errors": []}
    all_rows = []

    # 1) trending
    js, err = safe_get("https://api.dexscreener.com/latest/dex/trending")
    if err: meta["errors"].append(f"trending: {err}")
    pairs = js.get("pairs", []) if isinstance(js, dict) else []
    all_rows += rows_from_pairs(pairs, chains_selected)

    # 2) fallback search per chain if needed
    if not all_rows:
        meta["source"] = "dexscreener_search"
        queries = []
        if "Solana" in chains_selected:  queries.append("solana")
        if "Ethereum" in chains_selected: queries.append("ethereum")
        if "BSC" in chains_selected:      queries.append("bsc")
        if not queries:                   queries = ["solana","ethereum","bsc"]
        for q in queries:
            js2, err2 = safe_get(f"https://api.dexscreener.com/latest/dex/search?q={q}")
            if err2: meta["errors"].append(f"search({q}): {err2}")
            pairs2 = js2.get("pairs", []) if isinstance(js2, dict) else []
            all_rows += rows_from_pairs(pairs2, chains_selected)

    df = pd.DataFrame(all_rows)
    meta["rows]()
