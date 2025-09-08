# streamlit_app.py
import os, time, json, requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# ---------- Page ----------
st.set_page_config(page_title="Memecoin Dashboard", layout="wide")
st.markdown('''
<style>
.block-container {padding-top: 2rem;}
h1 {background: linear-gradient(90deg, #7C4DFF, #4DD0E1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
</style>
''', unsafe_allow_html=True)
st.title("🚀 Memecoin Dashboard — Best Build")
st.caption("Bigger demo, safer defaults, robust Live mode, diagnostics")
load_dotenv()

# ---------- Controls ----------
mode = st.radio("Data source mode", ["Demo (offline)", "Live (internet)"], horizontal=True)
chains = st.multiselect("Chains", ["Ethereum", "Solana", "BSC"], default=["Ethereum","Solana","BSC"])

is_demo = (mode == "Demo (offline)")
def_liq = 5_000 if is_demo else 50_000
def_vol = 1_000 if is_demo else 10_000
def_lock = False

colA, colB, colC, colD = st.columns(4)
with colA: min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=def_liq, step=1_000)
with colB: min_vol = st.number_input("Min 24h Volume ($)", min_value=0, value=def_vol, step=500)
with colC: max_age = st.number_input("Max Age (days) — 0 disables", min_value=0, value=180, step=1)
with colD: require_locked = st.checkbox("Require Liquidity Locked ≥ 70%", value=def_lock)

st.subheader("Weights (normalized)")
defaults = {"w_liq":20,"w_vol":20,"w_tx":10,"w_age":5,"w_lock":20,"w_top10":20,"w_sent":15,"w_security":10}
weights={}
labels=[("w_liq","Liquidity"),("w_vol","24h Volume"),("w_tx","Txns"),("w_age","Age (younger=better)"),
        ("w_lock","Liquidity Locked %"),("w_top10","Top-10 Holders % (lower better)"),("w_sent","Sentiment"),("w_security","Security")]
for i,(k,lab) in enumerate(labels):
    with st.columns(4)[i%4]:
        weights[k]=st.slider(lab,0,100,defaults[k],1)
wtot=sum(weights.values()) or 1
for k in weights: weights[k]=weights[k]/wtot

# ---------- Helpers ----------
def minmax(s, invert=False):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if s.max() == s.min(): norm = pd.Series(0.5, index=s.index)
    else: norm = (s - s.min()) / (s.max() - s.min())
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
    except: return 0.0

def to_int_txns_h24(tx):
    if tx is None: return 0
    if isinstance(tx, dict):
        h24 = tx.get("h24", 0)
        if isinstance(h24, dict):  # {'buys': X, 'sells': Y}
            total = 0
            for v in h24.values():
                try: total += int(v or 0)
                except: pass
            return total
        try: return int(h24 or 0)
        except: return 0
    try: return int(tx or 0)
    except: return 0

def safe_get(url, headers=None, params=None, timeout=15):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 200: return r.json(), None
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
        if chains_selected and chain not in chains_selected: continue
        rows.append(dict(
            name=(p.get("baseToken") or {}).get("name") or (p.get("baseToken") or {}).get("symbol") or "Unknown",
            symbol=(p.get("baseToken") or {}).get("symbol") or "?",
            chain=chain,
            price=to_float(p.get("priceUsd")),
            liquidity_usd=to_float((p.get("liquidity") or {}).get("usd", 0)),
            volume24h_usd=to_float((p.get("volume") or {}).get("h24", 0)),
            txns24h=to_int_txns_h24(p.get("txns")),
            mcap_usd=to_float(p.get("fdv")),
            age_days=np.nan,  # may enrich later
            is_honeypot=False, owner_renounced=False,
            liquidity_locked_pct=np.nan, top10_holders_pct=np.nan,
            telegram_members=np.nan, twitter_followers=np.nan, sentiment_score=np.nan
        ))
    return rows

# ---------- Data loaders ----------
@st.cache_data
def load_demo():
    return pd.read_csv("sample_data.csv")

@st.cache_data(ttl=180)
def load_live(chains_selected):
    meta = {"source": "dexscreener_trending", "errors": []}
    all_rows = []

    # 1) Trending
    js, err = safe_get("https://api.dexscreener.com/latest/dex/trending")
    if err: meta["errors"].append(f"trending: {err}")
    pairs = js.get("pairs", []) if isinstance(js, dict) else []
    all_rows += rows_from_pairs(pairs, chains_selected)

    # 2) Fallback: search per chain
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
    meta["rows_live"] = len(df)

    # 3) Optional Birdeye age for Solana symbols
    if not df.empty and "Solana" in df["chain"].unique():
        key = os.getenv("BIRDEYE_API_KEY", "").strip() or st.secrets.get("BIRDEYE_API_KEY", "")
        if key:
            headers = {"X-API-KEY": key}
            bj, e3 = safe_get(
                "https://public-api.birdeye.so/defi/tokenlist?sort_by=created&sort_type=desc&offset=0&limit=200",
                headers=headers
            )
            if e3: meta["errors"].append(f"birdeye: {e3}")
            tokens = (bj.get("data") or {}).get("tokens", []) if isinstance(bj, dict) else []
            bd = pd.DataFrame(tokens)
            if not bd.empty and {"symbol","createdTime"}.issubset(bd.columns):
                now = time.time()
                bd["age_days"] = (now - bd["createdTime"].astype(float)) / 86400.0
                sym_age = bd.groupby("symbol", as_index=False)["age_days"].min()
                mask_sol = df["chain"] == "Solana"
                df.loc[mask_sol, "age_days"] = df.loc[mask_sol].merge(sym_age, on="symbol", how="left")["age_days"].values

    df["age_days"] = df["age_days"].fillna(9999)  # unknown
    return df, meta

# ---------- Load ----------
meta = {"source": "demo", "errors": [], "rows_live": 0}
if mode == "Demo (offline)":
    try: data = load_demo()
    except Exception as e:
        st.error(f"Could not read sample_data.csv: {e}"); data = pd.DataFrame()
else:
    st.info("Fetching live data… (DexScreener; Birdeye optional for Solana age).")
    try: data, meta = load_live(chains)
    except Exception as e:
        meta["errors"].append(f"live_exception: {e}"); data = pd.DataFrame()

# Fallback to demo if live empty
if data.empty:
    try:
        data = load_demo()
        st.warning("Live fetch returned no rows. Showing Demo data so the app stays usable.")
        meta["source"] = "demo_fallback"
    except Exception as e:
        meta["errors"].append(f"demo_read: {e}")
        st.error("No data available (live and demo both failed)."); st.write("Diagnostics:", meta)

# Ensure columns exist
required_cols = ["name","symbol","chain","price","liquidity_usd","volume24h_usd","txns24h",
    "mcap_usd","age_days","is_honeypot","owner_renounced","liquidity_locked_pct",
    "top10_holders_pct","telegram_members","twitter_followers","sentiment_score"]
for c in required_cols:
    if c not in data.columns: data[c] = np.nan

# ---------- Filtering ----------
show_all_demo = st.checkbox("Show all demo rows (ignore filters)", value=is_demo)

def apply_filters(df_in):
    if show_all_demo: return df_in.copy()

    df1 = df_in[df_in["chain"].isin(chains)]
    df1 = df1[pd.to_numeric(df1["liquidity_usd"], errors="coerce").fillna(0) >= min_liq]
    df1 = df1[pd.to_numeric(df1["volume24h_usd"], errors="coerce").fillna(0) >= min_vol]

    # Age filter: 0 disables; unknown ages (NaN or 9999) are kept
    if max_age > 0:
        age = pd.to_numeric(df1["age_days"], errors="coerce")
        keep_unknown = age.isna() | (age >= 9999)
        keep_young  = age.le(max_age)
        df1 = df1[keep_unknown | keep_young]

    if require_locked and "liquidity_locked_pct" in df1.columns:
        df1 = df1[pd.to_numeric(df1["liquidity_locked_pct"], errors="coerce").fillna(0) >= 70]

    hide_honeypots = st.checkbox("Hide suspected honeypots", value=True)
    if "is_honeypot" in df1.columns and hide_honeypots:
        df1 = df1[~df1["is_honeypot"].fillna(False)]
    return df1

filtered = apply_filters(data)

# Percentile-based auto-relax (no age filter here)
relaxed_used = False
if not show_all_demo and filtered.empty and not data.empty:
    relaxed_used = True
    st.warning("No projects match your filters. Relaxing to dataset-based thresholds so you can see results.")
    def q(series, pct, fallback):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.quantile(pct)) if len(s) else fallback
    liq_p10 = q(data["liquidity_usd"], 0.10, 0)
    vol_p10 = q(data["volume24h_usd"], 0.10, 0)
    relaxed = data.copy()
    relaxed = relaxed[relaxed["chain"].isin(chains)]
    relaxed = relaxed[pd.to_numeric(relaxed["liquidity_usd"], errors="coerce").fillna(0) >= liq_p10]
    relaxed = relaxed[pd.to_numeric(relaxed["volume24h_usd"], errors="coerce").fillna(0) >= vol_p10]
    filtered = relaxed
    st.info({"auto_relax_thresholds": {"liquidity_min": round(liq_p10,2), "volume_min": round(vol_p10,2), "age_filter_applied": False}})

# ---------- Scoring ----------
scored = filtered.copy()
scored["score"]=(weights["w_liq"]*minmax(scored["liquidity_usd"]) +
                 weights["w_vol"]*minmax(scored["volume24h_usd"]) +
                 weights["w_tx"]*minmax(scored["txns24h"]) +
                 weights["w_age"]*(1-minmax(scored["age_days"])) +
                 weights["w_lock"]*minmax(scored["liquidity_locked_pct"]) +
                 weights["w_top10"]*(1-minmax(scored["top10_holders_pct"])) +
                 weights["w_sent"]*minmax(scored["sentiment_score"]) +
                 weights["w_security"]*(1-minmax(scored["is_honeypot"].fillna(False).astype(int)))).round(3)
ranked = scored.sort_values("score", ascending=False).reset_index(drop=True)

# ---------- Diagnostics ----------
with st.expander("🧪 Data Status / Diagnostics", expanded=False):
    st.write({
        "mode": mode,
        "data_source": meta.get("source"),
        "live_rows": meta.get("rows_live", None),
        "after_filter_rows": int(ranked.shape[0]),
        "relaxed_filters_used": relaxed_used,
        "errors": meta.get("errors", []),
        "by_chain": data["chain"].value_counts(dropna=False).to_dict() if not data.empty else {},
    })
    if st.checkbox("Show raw rows (head)"):
        st.dataframe(data.head(50), use_container_width=True)

# ---------- Table ----------
st.markdown("### Ranked Results")
if ranked.empty:
    st.error("No rows to display. Lower thresholds, set Max Age to 0, or show all demo rows.")
else:
    st.dataframe(
        ranked[["score","name","symbol","chain","price","liquidity_usd","volume24h_usd","txns24h",
                "mcap_usd","age_days","liquidity_locked_pct","top10_holders_pct","sentiment_score"]],
        use_container_width=True
    )

# ---------- Visuals ----------
st.markdown("### 📊 Visual Overview")
tab1, tab2, tab3 = st.tabs(["Overview", "Details", "Notes"])

with tab1:
    if ranked.empty:
        st.info("No data for charts.")
    else:
        top3 = ranked.head(3)
        cols = st.columns(3)
        for i, (_, r) in enumerate(top3.iterrows()):
            with cols[i]:
                st.subheader(f"{r['name']} ({r['symbol']})")
                st.metric("Score", f"{r['score']:.3f}")
                st.metric("Liquidity", f"${float(r['liquidity_usd']):,.0f}")
                st.metric("24h Volume", f"${float(r['volume24h_usd']):,.0f}")
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
    if ranked.empty:
        st.info("No selection available.")
    else:
        options = ranked["name"].tolist()
        sel = st.selectbox("Pick a project", options)
        if sel not in options: sel = options[0]
        row = ranked.loc[ranked["name"] == sel]
        if row.empty: row = ranked.iloc[[0]]
        row = row.iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", row["score"])
            st.metric("Price", f"${float(row['price'] or 0):.10f}")
        with col2:
            st.metric("Liquidity", f"${float(row['liquidity_usd']):,.0f}")
            st.metric("Mcap", f"${float(row['mcap_usd']):,.0f}")
        with col3:
            st.metric("Age", int(float(row['age_days'] or 0)))
            st.metric("Txns24h", int(row['txns24h'] or 0))

        def _norm(col, invert=False):
            return float(minmax(ranked[col], invert).loc[ranked["name"] == sel])
        radar_vals = {"Liquidity": _norm("liquidity_usd"),
                      "Volume": _norm("volume24h_usd"),
                      "Txns": _norm("txns24h"),
                      "Lock%": _norm("liquidity_locked_pct"),
                      "Top10(↓)": _norm("top10_holders_pct", invert=True),
                      "Sentiment": _norm("sentiment_score")}
        theta = list(radar_vals.keys())
        rvals = list(radar_vals.values()) + [list(radar_vals.values())[0]]
        fig = go.Figure(data=[go.Scatterpolar(r=rvals, theta=theta + [theta[0]],
                                              fill='toself', name=sel)])
        fig.update_layout(title="Attribute Radar",
                          polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("Use this tab as your trading journal — jot down entry/exit notes, catalysts, risks, etc.")
    if not ranked.empty:
        st.download_button("Export current table (CSV)",
            ranked.to_csv(index=False).encode("utf-8"),
            file_name="memecoin_ranked_export.csv", mime="text/csv")
