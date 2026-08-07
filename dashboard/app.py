import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="TSF Analytics", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

# ================= LOAD CSS TERPISAH =================
def load_css():
    here = Path(".")
    try:
        here = Path(__file__).parent
    except Exception:
        pass
    candidates = [
        Path("dashboard/style.css"),
        here / "style.css",
        Path("dashboard/styles.css"),
        Path("dashboard/style.css.txt"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue
    return ""

st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

# ================= PATHS =================
RAW_DIR = Path("data/raw")
FORECAST_PATH = Path("data/forecasts/forecast.csv")
METRICS_PATH = Path("models/metrics.csv")
EVAL_PATH = Path("data/forecasts/evaluation.csv")
CONC_PATH = Path("data/forecasts/conclusions.csv")

DOMAIN_COLORS = {"crypto": "#8b5cf6", "stock": "#6366f1", "energy": "#22d3ee", "food": "#f59e0b"}
SIGNAL_COLORS = {"UP": "#4ade80", "DOWN": "#f87171", "FLAT": "#9ca3af"}

def _f(v):
    try: return f"{float(v):,.2f}"
    except Exception: return "-"

def md_to_html(text):
    """Konversi markdown laporan (####, **, bullet) menjadi HTML rapi."""
    out = []
    for ln in str(text).split("\n"):
        if ln.startswith("#### "):
            out.append(f"<h4 style='margin:16px 0 6px 0;color:var(--text);'>{ln[5:]}</h4>")
        elif ln.strip():
            ln = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ln)   # <-- SATU backslash!
            if ln.startswith("* ") or ln.startswith("- "):
                ln = "<span style='color:var(--accent);margin-right:8px;'>•</span>" + ln[2:]
            out.append(f"<div style='margin:2px 0;'>{ln}</div>")
    return "".join(out)

# ================= DATA LOADING =================
@st.cache_data(ttl=60)
def load_raw():
    frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in sorted(RAW_DIR.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(ttl=60)
def load_fc():
    return pd.read_csv(FORECAST_PATH, parse_dates=["timestamp", "created_at"]) if FORECAST_PATH.exists() else pd.DataFrame()

@st.cache_data(ttl=60)
def load_ev():
    return pd.read_csv(EVAL_PATH, parse_dates=["timestamp"]) if EVAL_PATH.exists() else pd.DataFrame()

@st.cache_data(ttl=60)
def load_mt():
    return pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()

@st.cache_data(ttl=60)
def load_cc():
    return pd.read_csv(CONC_PATH, parse_dates=["created_at"]) if CONC_PATH.exists() else pd.DataFrame()

def latest_fc(fc):
    return fc.sort_values("created_at").groupby("series_id").tail(1) if not fc.empty else fc

def is_anom(s):
    return s.astype(str).str.lower().eq("true")

def base_layout(fig, title=None, legend=False):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#cbd5e1", size=11), margin=dict(l=8, r=8, t=36, b=8),
                      showlegend=legend, hovermode="x unified",
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1f2637"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color="#e5e7eb")))
    return fig

def filter_tf(df, tf):
    if tf == "ALL" or df.empty: return df
    days = {"7D": 7, "30D": 30, "90D": 90}[tf]
    return df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=days)]

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;margin:6px 0 22px 4px;">
    <div style="width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#8b5cf6,#6366f1);
    display:flex;align-items:center;justify-content:center;font-size:18px;">⚡</div>
    <div><b style="font-size:1.05rem;">TSF Analytics</b><br><span class="muted" style="font-size:.72rem;">Unified Forecasting</span></div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigasi", ["🏠 Overview", "📈 Forecast & Interval", "📐 Kinerja Model", "🤖 Laporan AI", "🗄️ Data Historis"],
                    label_visibility="collapsed", key="nav")

    st.markdown("---")
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;padding:6px 4px;">
    <div style="width:34px;height:34px;border-radius:50%;background:#232a3d;display:flex;align-items:center;justify-content:center;">👤</div>
    <div><b style="font-size:.85rem;">Admin</b><br><span class="muted" style="font-size:.7rem;">Market Monitor</span></div></div>""",
                unsafe_allow_html=True)

# ================= LOAD ALL DATA =================
raw, fc, ev, mt, cc = load_raw(), load_fc(), load_ev(), load_mt(), load_cc()
lat = latest_fc(fc)

# FIX: buang series usang (mis. CHILI_JKT/RICE_JKT yang sudah diarsipkan) agar konsisten 8 series
if not raw.empty and not lat.empty:
    lat = lat[lat["series_id"].isin(raw["series_id"].unique())]

n_anom = int(is_anom(lat["warning"]).sum()) if not lat.empty else 0

# ================= OVERVIEW =================
if page == "🏠 Overview":
    c = st.columns([3, 1])
    with c[0]:
        st.markdown("### Selamat datang kembali! 👋")
        st.markdown("<span class='muted'>Berikut kondisi pasar lintas domain secara real-time.</span>", unsafe_allow_html=True)
    with c[1]:
        st.markdown(f"<div style='text-align:right;color:var(--muted);font-size:.85rem;'>📅 {pd.Timestamp.now():%d %b %Y • %H:%M}</div>", unsafe_allow_html=True)

    k = st.columns(4)
    mape_avg = mt["mape_pct"].mean() if not mt.empty else np.nan
    cov = ev["within_80"].dropna().mean() * 100 if not ev.empty and "within_80" in ev.columns else np.nan
    k[0].metric("Series Dipantau", len(raw["series_id"].unique()) if not raw.empty else 0, f"{raw['domain'].nunique()} domain")
    k[1].metric("Anomali Aktif", n_anom, "normal" if n_anom == 0 else "perlu perhatian", delta_color="inverse")
    k[2].metric("MAPE Rata-rata", f"{mape_avg:.2f}%", "validasi")
    k[3].metric("Coverage Live", f"{cov:.0f}%" if pd.notna(cov) else "-", "target 80%")

    g = st.columns([2, 1], gap="medium")
    with g[0]:
        st.markdown("<div class='card'><div class='card-title'>Market Pulse</div></div>", unsafe_allow_html=True)
        cc1, cc2 = st.columns([3, 1])
        sel = cc1.selectbox("Series", sorted(raw["series_id"].unique()), label_visibility="collapsed")
        tf = cc2.radio("Range", ["30D", "90D", "ALL"], horizontal=True, label_visibility="collapsed")
        d = filter_tf(raw[raw["series_id"] == sel], tf)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["value"], mode="lines", name=sel,
                                 line=dict(color="#8b5cf6", width=2.2), fill="tozeroy",
                                 fillcolor="rgba(139,92,246,0.16)"))
        if not d.empty:
            lo, hi = d["value"].min(), d["value"].max()
            pad = (hi - lo) * 0.15 if hi > lo else hi * 0.05
            fig.update_yaxes(range=[lo - pad, hi + pad])
        st.plotly_chart(base_layout(fig), use_container_width=True)
    with g[1]:
        st.markdown("<div class='card'><div class='card-title'>Komposisi Domain</div></div>", unsafe_allow_html=True)
        counts = raw.groupby("domain")["series_id"].nunique() if not raw.empty else pd.Series(dtype=int)
        total_series = int(counts.sum())
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.68, showlegend=False,
                               marker=dict(colors=[DOMAIN_COLORS.get(i, "#8b5cf6") for i in counts.index],
                                           line=dict(color="#0b0e17", width=2))))
        fig.add_annotation(text=f"<b>{total_series}</b><br><span style='font-size:10px'>series</span>",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="#e5e7eb"))
        st.plotly_chart(base_layout(fig), use_container_width=True)

        legend_parts = []
        for d_name, n in counts.items():
            color = DOMAIN_COLORS.get(d_name, "#8b5cf6")
            pct = n / total_series * 100 if total_series else 0
            legend_parts.append(
                f"<div class='feed-row'><span class='feed-dot' style='background:{color}'></span>"
                f"<span>{d_name}</span><span class='muted' style='margin-left:auto'>{n} series ({pct:.0f}%)</span></div>"
            )
        st.markdown("".join(legend_parts), unsafe_allow_html=True)

    t = st.columns(3, gap="medium")
    with t[0]:
        st.markdown("<div class='card'><div class='card-title'>🚨 Peringatan Dini</div></div>", unsafe_allow_html=True)
        act = lat[is_anom(lat["warning"])] if not lat.empty else lat
        if act is not None and not act.empty:
            for _, r in act.iterrows():
                dev = _f(r.get("deviation_pct"))
                sid = r["series_id"]
                st.markdown(f"""<div class='feed-row'><span class='feed-dot' style='background:var(--amber)'></span>
                <div><b>{sid}</b><br><span class='muted'>deviasi {dev}% dari pola normal</span></div></div>""",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div class='feed-row'><span class='feed-dot' style='background:var(--green)'></span>Tidak ada anomali aktif.</div>", unsafe_allow_html=True)
    with t[1]:
        st.markdown("<div class='card'><div class='card-title'>🏆 Top Movers (Prediksi)</div></div>", unsafe_allow_html=True)
        if not lat.empty and "change_pct" in lat.columns:
            top = lat.reindex(lat["change_pct"].abs().sort_values(ascending=False).index).head(5)
            rows = ""
            for _, r in top.iterrows():
                ch = r["change_pct"] if pd.notna(r.get("change_pct")) else 0
                col = "#4ade80" if ch > 0 else ("#f87171" if ch < 0 else "#9ca3af")
                arrow = "▲" if ch > 0 else ("▼" if ch < 0 else "•")
                dom = r["domain"]
                sid = r["series_id"]
                rows += f"""<div class='feed-row'><span class='badge b-{dom}'>{dom}</span>
                <span><b>{sid}</b></span>
                <span style='margin-left:auto;color:{col}'>{arrow} {abs(ch):.2f}%</span></div>"""
            st.markdown(rows, unsafe_allow_html=True)
    with t[2]:
        st.markdown("<div class='card'><div class='card-title'>⚙️ Aktivitas Sistem</div></div>", unsafe_allow_html=True)
        if not fc.empty:
            for _, r in fc.sort_values("created_at", ascending=False).head(5).iterrows():
                sig = r["signal"]
                color_sig = SIGNAL_COLORS.get(sig, "#9ca3af")
                sid = r["series_id"]
                pred_val = _f(r["predicted_next_value"])
                time_str = pd.Timestamp(r["created_at"]).strftime("%H:%M")
                st.markdown(f"""<div class='feed-row'><span class='feed-dot' style='background:{color_sig}'></span>
                <div><b>{sid}</b> <span class='muted'>• {sig} • {pred_val}</span></div>
                <span class='muted' style='margin-left:auto;font-size:.72rem;'>{time_str}</span></div>""",
                            unsafe_allow_html=True)

# ================= FORECAST & INTERVAL =================
elif page == "📈 Forecast & Interval":
    st.markdown("### Forecast & Interval Keyakinan 80%")
    doms = st.multiselect("Filter domain", sorted(raw["domain"].unique()), default=sorted(raw["domain"].unique()))
    sub = lat[lat["domain"].isin(doms)] if not lat.empty else lat
    st.dataframe(sub[["series_id", "domain", "last_value", "predicted_next_value", "lower_bound", "upper_bound", "change_pct", "signal", "warning"]]
                 .sort_values("series_id"), use_container_width=True, hide_index=True)

    sel = st.selectbox("Detail series", sorted(raw["series_id"].unique()))
    d = raw[raw["series_id"] == sel].sort_values("timestamp")
    fb = fc[(fc["series_id"] == sel) & fc["lower_bound"].notna()].sort_values("timestamp") if not fc.empty else fc
    fig = go.Figure()
    if fb is not None and not fb.empty:
        fig.add_trace(go.Scatter(x=fb["timestamp"], y=fb["upper_bound"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fb["timestamp"], y=fb["lower_bound"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(139,92,246,0.18)", showlegend=False, name="Interval 80%"))
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["value"], mode="lines", name="Aktual", line=dict(color="#64748b", width=1.4)))
    if fb is not None and not fb.empty:
        fig.add_trace(go.Scatter(x=fb["timestamp"], y=fb["predicted_next_value"],
                                 mode="lines+markers", name="Prediksi", line=dict(color="#8b5cf6", width=2)))
    st.plotly_chart(base_layout(fig, title=f"{sel} — aktual vs prediksi + interval", legend=True), use_container_width=True)

# ================= KINERJA MODEL =================
elif page == "📐 Kinerja Model":
    st.markdown("### Kinerja Model")
    st.markdown("<div class='card'><div class='card-title'>Metrik Validasi</div></div>", unsafe_allow_html=True)
    if not mt.empty:
        cols = [c for c in ["series_id", "mae", "rmse", "mape_pct", "n_train", "n_valid"] if c in mt.columns]
        st.dataframe(mt[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Metrik belum ada. Jalankan training dulu.")

    st.markdown("<div class='card'><div class='card-title'>Track Record Live (Prediksi vs Aktual)</div></div>", unsafe_allow_html=True)
    if not ev.empty:
        summ = ev.groupby("series_id").agg(
            n_eval=("abs_error", "size"),
            mae=("abs_error", "mean"),
            mape_pct=("ape_pct", "mean"),
            coverage_pct=("within_80", lambda s: round(s.dropna().mean() * 100, 1) if s.notna().any() else None),
        ).round(2).reset_index()
        st.dataframe(summ, use_container_width=True, hide_index=True)
        st.caption("`coverage_pct` = seberapa sering aktual masuk interval 80% (ideal ≈ 80). `n_eval` = jumlah prediksi yang sudah terbukti.")

        a, b = st.columns([1, 1], gap="medium")
        with a:
            st.markdown("<div class='card'><div class='card-title'>MAPE per Series (validasi)</div></div>", unsafe_allow_html=True)
            if not mt.empty:
                fig = go.Figure(go.Bar(x=mt["mape_pct"], y=mt["series_id"], orientation="h", marker=dict(color="#8b5cf6")))
                st.plotly_chart(base_layout(fig), use_container_width=True)
        with b:
            st.markdown("<div class='card'><div class='card-title'>Coverage Live per Series</div></div>", unsafe_allow_html=True)
            fig2 = go.Figure(go.Bar(x=summ["coverage_pct"], y=summ["series_id"], orientation="h", marker=dict(color="#22d3ee")))
            st.plotly_chart(base_layout(fig2), use_container_width=True)
    else:
        st.info("Evaluasi belum ada. Jalankan: python -m src.evaluate")

# ================= LAPORAN AI =================
elif page == "🤖 Laporan AI":
    st.markdown("### Analisis AI — Laporan Pasar")
    if not cc.empty:
        last = cc.iloc[-1]
        engine = last["engine"]
        created = pd.Timestamp(last["created_at"]).strftime("%d %b %Y, %H:%M")
        st.markdown(f"<span class='badge b-crypto'>engine: {engine}</span> "
                    f"<span class='muted' style='font-size:.8rem;'>• {created}</span>",
                    unsafe_allow_html=True)
        # FIX: render markdown menjadi HTML rapi (heading + bold)
        st.markdown(f"<div class='card'>{md_to_html(last['conclusion'])}</div>", unsafe_allow_html=True)
        with st.expander("Riwayat laporan"):
            for _, r in cc.sort_values("created_at", ascending=False).head(10).iterrows():
                ts = pd.Timestamp(r["created_at"]).strftime("%d %b %H:%M")
                st.markdown(f"**{ts}** • {r['engine']}")
                st.caption(str(r["conclusion"]).replace("#### ", "").split("\n")[0])
                st.markdown("---")
    else:
        st.info("Kesimpulan belum ada. Jalankan: python -m src.analyst")

# ================= DATA HISTORIS =================
elif page == "🗄️ Data Historis":
    st.markdown("### Data Historis")
    tf = st.radio("Range", ["30D", "90D", "ALL"], horizontal=True)
    for sid, grp in raw.groupby("series_id"):
        dom = grp["domain"].iloc[0]
        color = DOMAIN_COLORS.get(dom, "#8b5cf6")
        st.markdown(f"<span class='badge b-{dom}'>{dom}</span> <b>{sid}</b>", unsafe_allow_html=True)
        d = filter_tf(grp, tf)
        fig = go.Figure(go.Scatter(x=d["timestamp"], y=d["value"], mode="lines",
                                   line=dict(color=color, width=1.6)))
        st.plotly_chart(base_layout(fig), use_container_width=True)

    with st.expander("Lihat tabel data lengkap"):
        st.dataframe(raw.tail(50), use_container_width=True, hide_index=True)