# streamlit_app.py
import os, math, json, time
import pandas as pd
import numpy as np
import requests
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

load_dotenv()  # loads .env locally; in Streamlit Cloud use Secrets

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

def std_cols():
    """Return empty DataFrame with the standard schema we use."""
    return pd.DataFrame(columns=[
        "name","symbol","chain","price","liquidity_usd","volume24h_usd","txns24h",
        "mcap_usd","age_days","is_honeypot","owner_renounced","liquidity_locked_pct",
        "top10_holders_pct","telegram_members","twitter_followers","sentiment_score"
    ])

def normalize_chain(cid: str) -> str:
    cid = (cid or "").lower()
    if cid in ("eth","ethereum"): return "Ethereum"
    if cid in ("sol","solana"):   return "Solana"
    if cid in ("bsc","bnb"):      return "BSC"
    return cid.capitalize() if cid else "Unknown"

def safe_get(url, headers=None, params=None, timeout=15):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# -------------------- Data loaders --------------------
@st.cache_data
def load_demo():
    return pd.read_csv("sample_data.csv")

@st.cache_data(ttl=180)
def load_live(chains_selected):
    """
    Live mode:
      - DexScreener trending pairs (no key) → base liquidity/volume/txns/mcap/price
      - If BIRDEYE_API_KEY set, enrich SOL tokens with created time (age)
    Returns DataFrame with our standard columns.
    """
    rows = []
    # DexScreener trending
    ds = safe_get("https://api.dexscreener.com/latest/dex/trending") or {}
    pairs = ds.get("pairs", []) if isinstance(ds, dict) else []

    for p in pairs[:200]:
        chain = normalize_chain(p.get("chainId", ""))
        if chain not in chains_selected:  # respect filter early
            continue

        liq_usd = (p.get("liquidity") or {}).get("usd") or 0
        vol24  = (p.get("volume") or {}).get("h24") or 0
        tx24   = (p.get("txns") or {}).get("h24") or 0
        fdv    = p.get("fdv") or 0
        price  = p.get("priceUsd") or 0

        base = p.get("baseToken") or {}
        name = base.get("name") or base.get("symbol") or "Unknown"
        sym  = base.get("symbol") or "?"

        # placeholders; can be enriched later
        rows.append(dict(
            name=name, symbol=sym, chain=chain,
            price=float(price or 0),
            liquidity_usd=float(liq_usd or 0),
            volume24h_usd=float(vol24 or 0),
            txns24h=int(tx24 or 0),
            mcap_usd=float(fdv or 0),
            age_days=np.nan,                # will fill for Solana via Birdeye if possible
            is_honeypot=False,              # needs security API (GoPlus) if you add it
            owner_renounced=False,
            liquidity_locked_pct=np.nan,    # needs locker/explorer API
            top10_holders_pct=np.nan,       # needs explorer API
            telegram_members=np.nan,
            twitter_followers=np.nan,
            sentiment_score=np.nan
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        return std_cols()

    # Optional Birdeye enrichment (Solana age)
    birdeye_key = os.getenv("BIRDEYE_API_KEY", "").strip() or st.secrets.get("BIRDEYE_API_KEY", "")
    if birdeye_key and ("Solana" in df["chain"].unique()):
        # Endpoint: tokens created recently (we'll just pull some and map by symbol/name best-effort)
        # You can replace with a more precise endpoint if you have contract addresses.
        headers = {"X-API-KEY": birdeye_key}
        # Try recent tokens list
        bj = safe_get(
            "https://public-api.birdeye.so/defi/tokenlist?sort_by=created&sort_type=desc&offset=0&limit=200",
            headers=headers
        ) or {}
        tokens = (bj.get("data") or {}).get("tokens", []) if isinstance(bj, dict) else []
        bd = pd.DataFrame(tokens)
        # Map heuristically by symbol/name (imperfect but useful for rough age)
        if not bd.empty and "symbol" in bd.columns and "createdTime" in bd.columns:
            # createdTime is epoch seconds
            now = time.time()
            bd["age_days"] = (now - bd["createdTime"].astype(float)) / 86400.0
            # prefer exact symbol matches
            sym_age = bd.groupby("symbol", as_index=False)["age_days"].min()
            # apply where chain is Solana
            mask_sol = df["chain"] == "Solana"
            df.loc[mask_sol, "age_days"] = df.loc[mask_sol].merge(
                sym_age, left_on="symbol", right_on="symbol", how="left"
            )["age_days"].values

    # Fill unknown ages with a big number so the "Age (younger=better)" transform still works
    df["age_days"] = df["age_days"].fillna(9999)

    return df

# -------------------- Load data based on mode --------------------
if mode == "Demo (offline)":
    data = load_demo()
else:
    st.info("Fetching live data… (DexScreener; Birdeye optional for Solana age).")
    data = load_live(chains)
    if data.empty:
        st.warning("Live fetch returned no rows. Falling back to Demo.")
        data = load_demo()

# Ensure required columns exist
required_cols = [
    "name","symbol","chain","price","liquidity_usd","volume24h_usd","txns24h",
    "mcap_usd","age_days","is_honeypot","owner_renounced","liquidity_locked_pct",
    "top10_holders_pct","telegram_members","twitter_followers","sentiment_score"
]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"Missing columns in dataset: {missing}")
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
    st.warning("No projects match your filters. Relaxing thresholds so you can see results.")
    relaxed = df.copy()
    relaxed = relaxed[relaxed["chain"].isin(chains)]
    relaxed = relaxed[relaxed["liquidity_usd"] >= max(0, min_liq * 0.25)]
    relaxed = relaxed[relaxed["volume24h_usd"] >= max(0, min_vol * 0.25)]
    relaxed = relaxed[relaxed["age_days"] <= (max_age if max_age > 0 else relaxed["age_days"].max())]
    filtered = relaxed

if filtered.empty:
    st.error("Still no data after relaxing. Try lowering min liquidity/volume, or unchecking lock requirement.")
    st.stop()

# -------------------- Scoring --------------------
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
