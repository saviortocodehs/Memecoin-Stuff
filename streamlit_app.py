# streamlit_app.py
import os, time, json, requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# ---------- Page ----------
st.set_page_config(page_title="Memecoin Advisor — Pump.fun + DEX", layout="wide")
st.markdown('''
<style>
.block-container {padding-top: 2rem;}
h1 {background: linear-gradient(90deg, #7C4DFF, #4DD0E1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.small {font-size: 0.85rem; opacity: 0.9}
.badge {padding: 2px 8px; border-radius: 9999px; font-weight: 700; white-space: nowrap}
.badge-green {background:#16a34a22; color:#16a34a}
.badge-yellow{background:#ca8a0422; color:#ca8a04}
.badge-red{background:#dc262622; color:#dc2626}
</style>
''', unsafe_allow_html=True)

st.title("🚦 Memecoin Advisor — Signals & Scores (Pump.fun + DEX)")
st.caption("Pump.fun mode • Live DEX mode • Demo • Dedupe • Verdicts • Diagnostics")
load_dotenv()

# ---------- Controls ----------
mode = st.radio(
    "Data source mode",
    ["Demo (offline)", "Live (DexScreener)", "Pump.fun"],
    horizontal=True
)
chains = st.multiselect("Chains (DEX mode only)", ["Ethereum", "Solana", "BSC"],
                        default=["Ethereum","Solana","BSC"], disabled=(mode!="Live (DexScreener)"))

is_demo = (mode == "Demo (offline)")
def_liq = 5_000 if is_demo or mode=="Pump.fun" else 50_000
def_vol = 1_000 if is_demo or mode=="Pump.fun" else 10_000
def_lock = False

colA, colB, colC, colD = st.columns(4)
with colA: min_liq = st.number_input("Min Liquidity ($)", min_value=0, value=def_liq, step=1_000)
with colB: min_vol = st.number_input("Min 24h Volume ($)", min_value=0, value=def_vol, step=500)
with colC: max_age = st.number_input("Max Age (days) — 0 disables", min_value=0, value=180, step=1)
with colD: require_locked = st.checkbox("Require Liquidity Locked ≥ 70%", value=def_lock)

st.subheader("Weights (normalized)")
defaults = {"w_liq":18,"w_vol":22,"w_tx":12,"w_age":8,"w_lock":18,"w_top10":14,"w_sent":6,"w_security":2}
weights={}
labels=[("w_liq","Liquidity"),("w_vol","24h Volume"),("w_tx","Txns"),
        ("w_age","Age (younger=better)"),("w_lock","Liquidity Locked %"),
        ("w_top10","Top-10 Holders % (lower better)"),("w_sent","Sentiment"),("w_security","Security")]
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
        if isinstance(h24, dict):
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

def pick_row_by_name(ranked_df: pd.DataFrame, name_value: str):
    if ranked_df.empty: return pd.Series(dtype="object"), -1
    matched = ranked_df.index[ranked_df["name"] == name_value]
    idx = int(matched[0]) if len(matched) else int(ranked_df.index[0])
    return ranked_df.loc[idx], idx

# ---------- Row builders ----------
def rows_from_pairs(pairs, chains_selected):
    rows = []
    for p in pairs:
        chain = normalize_chain(p.get("chainId", ""))
        if chains_selected and chain not in chains_selected: continue
        base = p.get("baseToken") or {}
        rows.append(dict(
            name=base.get("name") or base.get("symbol") or "Unknown",
            symbol=base.get("symbol") or "?",
            chain=chain,
            base_address=base.get("address") or "",
            pair_url=p.get("url") or "",
            dex_id=p.get("dexId") or "",
            price=to_float(p.get("priceUsd")),
            liquidity_usd=to_float((p.get("liquidity") or {}).get("usd", 0)),
            volume24h_usd=to_float((p.get("volume") or {}).get("h24", 0)),
            txns24h=to_int_txns_h24(p.get("txns")),
            mcap_usd=to_float(p.get("fdv")),
            age_days=np.nan,
            is_honeypot=False, owner_renounced=False,
            liquidity_locked_pct=np.nan, top10_holders_pct=np.nan,
            telegram_members=np.nan, twitter_followers=np.nan, sentiment_score=np.nan
        ))
    return rows

def rows_from_pumpfun(coins):
    rows = []
    now = time.time()
    for c in coins:
        created_ts = c.get("createdTimestamp")
        try:
            age_days = (now - float(created_ts)) / 86400.0 if created_ts else 9999
        except Exception:
            age_days = 9999
        rows.append(dict(
            name=c.get("name") or c.get("symbol") or "Unknown",
            symbol=c.get("symbol") or "?",
            chain="Solana",
            base_address=c.get("mint") or "",
            pair_url=f"https://pump.fun/{c.get('mint')}" if c.get("mint") else "",
            dex_id="pumpfun",
            price=to_float(c.get("usdPrice")),
            liquidity_usd=to_float(c.get("liquidity")),
            volume24h_usd=to_float(c.get("volume24h")),
            txns24h=to_int_txns_h24(c.get("txns")),
            mcap_usd=to_float(c.get("marketCap")),
            age_days=age_days,
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
def load_live_dex(chains_selected):
    meta = {"source": "dexscreener_trending", "errors": []}
    all_rows = []
    js, err = safe_get("https://api.dexscreener.com/latest/dex/trending")
    if err: meta["errors"].append(f"trending: {err}")
    pairs = js.get("pairs", []) if isinstance(js, dict) else []
    all_rows += rows_from_pairs(pairs, chains_selected)
    if not all_rows:
        meta["source"] = "dexscreener_search"
        queries = []
        if "Solana" in chains_selected:  queries.append("solana")
        if "Ethereum" in chains_selected: queries.append("ethereum")
        if "BSC" in chains_selected:      queries.append("bsc")
        if not queries: queries = ["solana","ethereum","bsc"]
        for q in queries:
            js2, err2 = safe_get(f"https://api.dexscreener.com/latest/dex/search?q={q}")
            if err2: meta["errors"].append(f"search({q}): {err2}")
            pairs2 = js2.get("pairs", []) if isinstance(js2, dict) else []
            all_rows += rows_from_pairs(pairs2, chains_selected)
    df = pd.DataFrame(all_rows)
    meta["rows_live"] = len(df)
    df["age_days"] = df["age_days"].fillna(9999)
    return df, meta

@st.cache_data(ttl=60)
def load_pumpfun():
    # Common endpoints: trending / new / active
    urls = [
        "https://frontend-api.pump.fun/coins/trending",
        # Uncomment if you want more sources mixed in:
        # "https://frontend-api.pump.fun/coins/new",
        # "https://frontend-api.pump.fun/coins/active"
    ]
    all_rows = []
    meta = {"source":"pumpfun", "errors":[]}
    for url in urls:
        js, err = safe_get(url)
        if err:
            meta["errors"].append(f"{url}: {err}")
            continue
        coins = js if isinstance(js, list) else js.get("coins", [])
        all_rows += rows_from_pumpfun(coins)
    df = pd.DataFrame(all_rows)
    meta["rows_live"] = len(df)
    return df, meta

# ---------- Load ----------
meta = {"source": "demo", "errors": [], "rows_live": 0}
if mode == "Demo (offline)":
    try: data = load_demo()
    except Exception as e:
        st.error(f"Could not read sample_data.csv: {e}"); data = pd.DataFrame()
elif mode == "Pump.fun":
    st.info("Fetching Pump.fun tokens…")
    try: data, meta = load_pumpfun()
    except Exception as e:
        meta["errors"].append(f"pumpfun_exception: {e}"); data = pd.DataFrame()
else:  # Live Dex
    st.info("Fetching live DEX data… (DexScreener)")
    try: data, meta = load_live_dex(chains)
    except Exception as e:
        meta["errors"].append(f"dex_exception: {e}"); data = pd.DataFrame()

# Fallback to demo if empty
if data.empty:
    try:
        data = load_demo()
        st.warning("Primary fetch returned no rows. Showing Demo data so the app stays usable.")
        meta["source"] = "demo_fallback"
    except Exception as e:
        meta["errors"].append(f"demo_read: {e}")
        st.error("No data available (primary and demo both failed)."); st.write("Diagnostics:", meta)

# ---------- Cleanup: dedupe & generic names ----------
if not data.empty:
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    data["name"]   = data["name"].astype(str).str.strip()
    bad_names = {"SOLANA","ETHEREUM","BSC","BNB","ETH"}
    data = data[~data["name"].str.upper().isin(bad_names)]

    # Prefer the most active entry per token
    # For DEX data, use (chain, symbol). For Pump.fun (Solana-only), use base_address if available.
    group_keys = ["chain","symbol"]
    if mode == "Pump.fun" and "base_address" in data.columns and data["base_address"].notna().any():
        group_keys = ["chain","base_address"]
    if "volume24h_usd" in data.columns:
        try:
            best_idx = (
                data.groupby(group_keys)["volume24h_usd"]
                .idxmax()
                .dropna()
                .astype(int)
            )
            data = data.loc[best_idx].reset_index(drop=True)
        except Exception:
            data = data.reset_index(drop=True)

# Ensure columns exist
required_cols = ["name","symbol","chain","price","liquidity_usd","volume24h_usd","txns24h",
    "mcap_usd","age_days","is_honeypot","owner_renounced","liquidity_locked_pct",
    "top10_holders_pct","telegram_members","twitter_followers","sentiment_score",
    "base_address","pair_url","dex_id"]
for c in required_cols:
    if c not in data.columns: data[c] = np.nan

# ---------- Advisor rules (verdicts) ----------
def compute_verdict_row(r):
    """Return (verdict_str, verdict_score_0_10, reasons_list)."""
    liq   = float(pd.to_numeric(r.get("liquidity_usd"), errors="coerce") or 0)
    vol   = float(pd.to_numeric(r.get("volume24h_usd"), errors="coerce") or 0)
    tx    = float(pd.to_numeric(r.get("txns24h"), errors="coerce") or 0)
    lockp = float(pd.to_numeric(r.get("liquidity_locked_pct"), errors="coerce") if pd.notna(r.get("liquidity_locked_pct")) else -1)
    top10 = float(pd.to_numeric(r.get("top10_holders_pct"), errors="coerce") if pd.notna(r.get("top10_holders_pct")) else -1)
    honeypot = bool(r.get("is_honeypot") or False)
    age  = float(pd.to_numeric(r.get("age_days"), errors="coerce") or 9999)

    score = 0
    reasons = []

    # Liquidity
    if liq >= 50000: score += 2; reasons.append("✅ Liquidity ≥ $50k")
    elif liq >= 20000: score += 1; reasons.append("🟨 Liquidity ≥ $20k")
    else: reasons.append("❌ Very low liquidity")

    # Volume + Txns momentum
    if vol >= 25000 and tx >= 300: score += 2; reasons.append("✅ Healthy 24h volume & activity")
    elif vol >= 10000 and tx >= 100: score += 1; reasons.append("🟨 Moderate 24h volume/txns")
    else: reasons.append("❌ Weak demand")

    # Lock / renounce (if provided)
    if lockp >= 70: score += 2; reasons.append("✅ Liquidity locked ≥ 70%")
    elif 0 <= lockp < 70: reasons.append("🟨 Low lock%")
    else: reasons.append("🟨 Lock% unknown")

    # Whale distribution (if provided)
    if 0 <= top10 <= 25: score += 2; reasons.append("✅ Top10 holders ≤ 25%")
    elif 25 < top10 <= 50: score += 1; reasons.append("🟨 Top10 holders ≤ 50%")
    elif top10 > 50: reasons.append("❌ Concentrated holders")
    else: reasons.append("🟨 Holder distribution unknown")

    # Honeypot / security
    if honeypot: reasons.append("❌ Honeypot/suspicious"); score -= 3

    # Age (optional/soft)
    if 1 <= age <= 14: score += 1; reasons.append("🟩 Early but not newborn")
    elif age < 1: reasons.append("🟨 Newborn token (sniper risk)")

    if score >= 7: verdict = "✅ BUYABLE"
    elif score >= 4: verdict = "⚠️ WATCH"
    else: verdict = "❌ AVOID"
    return verdict, max(0, min(10, score)), reasons

# ---------- Filtering ----------
show_all_demo = st.checkbox("Show all demo rows (ignore filters)", value=is_demo)

def apply_filters(df_in):
    if show_all_demo: return df_in.copy()
    df1 = df_in.copy()
    if mode == "Live (DexScreener)":
        df1 = df1[df1["chain"].isin(chains)]
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

# Percentile auto-relax (no age)
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
    if mode == "Live (DexScreener)":
        relaxed = relaxed[relaxed["chain"].isin(chains)]
    relaxed = relaxed[pd.to_numeric(relaxed["liquidity_usd"], errors="coerce").fillna(0) >= liq_p10]
    relaxed = relaxed[pd.to_numeric(relaxed["volume24h_usd"], errors="coerce").fillna(0) >= vol_p10]
    filtered = relaxed
    st.info({"auto_relax_thresholds": {"liquidity_min": round(liq_p10,2), "volume_min": round(vol_p10,2), "age_filter_applied": False}})

# ---------- Scoring (numeric) ----------
scored = filtered.copy()
scored["score"]=(weights["w_liq"]*minmax(scored["liquidity_usd"]) +
                 weights["w_vol"]*minmax(scored["volume24h_usd"]) +
                 weights["w_tx"]*minmax(scored["txns24h"]) +
                 weights["w_age"]*(1-minmax(scored["age_days"])) +
                 weights["w_lock"]*minmax(scored["liquidity_locked_pct"]) +
                 weights["w_top10"]*(1-minmax(scored["top10_holders_pct"])) +
                 weights["w_sent"]*minmax(scored["sentiment_score"]) +
                 weights["w_security"]*(1-minmax(scored["is_honeypot"].fillna(False).astype(int)))).round(3)

# ---------- Verdicts ----------
verdicts, scores10, reasons_col = [], [], []
for _, r in scored.iterrows():
    v, s10, rs = compute_verdict_row(r)
    verdicts.append(v); scores10.append(s10); reasons_col.append(" • ".join(rs))
scored["signal_score_10"] = scores10
scored["verdict"] = verdicts
scored["reasons"] = reasons_col

# Sort by verdict class, then numeric score, then volume
verdict_rank = scored["verdict"].map({"✅ BUYABLE":2, "⚠️ WATCH":1, "❌ AVOID":0}).fillna(0)
ranked = scored.assign(_v=verdict_rank).sort_values(
    ["_v","score","volume24h_usd"], ascending=[False, False, False]
).drop(columns=["_v"]).reset_index(drop=True)

# ---------- Diagnostics ----------
with st.expander("🧪 Data Status / Diagnostics", expanded=False):
    st.write({
        "mode": mode, "data_source": meta.get("source"),
        "live_rows": meta.get("rows_live", None),
        "after_filter_rows": int(ranked.shape[0]),
        "relaxed_filters_used": relaxed_used,
        "errors": meta.get("errors", []),
        "by_chain": data["chain"].value_counts(dropna=False).to_dict() if not data.empty else {},
    })
    if st.checkbox("Show raw rows (head)"):
        st.dataframe(data.head(50), use_container_width=True)

# ---------- Table ----------
st.markdown("### Ranked Results with Verdicts")
if ranked.empty:
    st.error("No rows to display. Lower thresholds, set Max Age to 0, or show all demo rows.")
else:
    def badge(v):
        if "BUYABLE" in v: return f'<span class="badge badge-green">{v}</span>'
        if "WATCH"   in v: return f'<span class="badge badge-yellow">{v}</span>'
        return f'<span class="badge badge-red">{v}</span>'
    show = ranked.copy()
    show["verdict"] = show["verdict"].apply(badge)
    show["pair"] = show["pair_url"].apply(lambda u: f'<a href="{u}" target="_blank">Open</a>' if isinstance(u,str) and u else "")
    cols = ["verdict","signal_score_10","score","name","symbol","chain","price",
            "liquidity_usd","volume24h_usd","txns24h","mcap_usd","reasons","dex_id","pair","base_address"]
    st.write(show[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

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
                v = r["verdict"]
                color = "green" if "BUYABLE" in v else ("orange" if "WATCH" in v else "red")
                st.subheader(f"{r['name']} ({r['symbol']})")
                st.markdown(f'<span class="badge badge-{"green" if color=="green" else ("yellow" if color=="orange" else "red")}">{v}</span>', unsafe_allow_html=True)
                st.metric("Score", f"{r['score']:.3f}")
                st.metric("Signal Score (0–10)", f"{int(r['signal_score_10'])}")
                st.metric("Liquidity", f"${float(r['liquidity_usd']):,.0f}")
                st.metric("24h Volume", f"${float(r['volume24h_usd']):,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            top_vol = ranked.nlargest(10, "volume24h_usd")[["name","volume24h_usd","signal_score_10","verdict"]]
            fig_bar = px.bar(top_vol, x="name", y="volume24h_usd",
                             hover_data=["signal_score_10","verdict"], title="Top 10 by 24h Volume")
            fig_bar.update_layout(xaxis_title="", yaxis_title="24h Volume (USD)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_scatter = px.scatter(
                ranked.head(120),
                x="liquidity_usd", y="volume24h_usd",
                size="mcap_usd", color="signal_score_10",
                hover_name="name", title="Liquidity vs Volume (colored by Signal Score)"
            )
            fig_scatter.update_layout(xaxis_title="Liquidity (USD)", yaxis_title="24h Volume (USD)")
            st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    if ranked.empty:
        st.info("No selection available.")
    else:
        options = ranked["name"].astype(str).tolist()
        sel = st.selectbox("Pick a project", options)
        row, idx = pick_row_by_name(ranked, sel)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Verdict", row.get("verdict"))
            st.metric("Signal Score (0–10)", int(row.get("signal_score_10", 0)))
            st.metric("Price", f"${float(row.get('price', 0) or 0):.10f}")
        with col2:
            st.metric("Liquidity", f"${float(row.get('liquidity_usd', 0) or 0):,.0f}")
            st.metric("Mcap", f"${float(row.get('mcap_usd', 0) or 0):,.0f}")
        with col3:
            st.metric("Age", int(float(row.get('age_days', 0) or 0)))
            st.metric("Txns24h", int(float(row.get('txns24h', 0) or 0)))

        def _norm(col, invert=False):
            s_norm = minmax(ranked[col], invert)
            try: return float(s_norm.loc[idx])
            except Exception:
                v = pd.to_numeric(ranked.loc[idx, col], errors="coerce")
                return float(0.5 if pd.isna(v) else v)

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
        fig = go.Figure(data=[go.Scatterpolar(r=rvals, theta=theta + [theta[0]], fill='toself', name=row.get("name","-"))])
        fig.update_layout(title="Attribute Radar", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Why this verdict:**")
        st.markdown(f"<div class='small'>{row.get('reasons','')}</div>", unsafe_allow_html=True)
        if isinstance(row.get("pair_url"), str) and row.get("pair_url"):
            st.link_button("Open Pair / Pump.fun", row.get("pair_url"))

with tab3:
    st.markdown("Use this tab as your trading journal — jot down entry/exit notes, catalysts, risks, etc.")
    if not ranked.empty:
        st.download_button(
            "Export current table (CSV)",
            ranked.to_csv(index=False).encode("utf-8"),
            file_name="memecoin_ranked_verdicts.csv",
            mime="text/csv"
        )
