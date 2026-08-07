import os
import sys
import hashlib
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

FORECAST_PATH = Path("data/forecasts/forecast.csv")
EVAL_PATH = Path("data/forecasts/evaluation.csv")
METRICS_PATH = Path("models/metrics.csv")
OUTPUT_PATH = Path("data/forecasts/conclusions.csv")

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

SYSTEM_PROMPT = (
    "Kamu adalah Kepala Analis Pasar (Chief Market Analyst) pada sistem forecasting multi-domain "
    "(kripto, saham, energi, pangan). Tulis LAPORAN ANALISIS dalam Bahasa Indonesia dengan gaya "
    "riset institusional: objektif, padat, berbasis data.\n"
    "SUMBER KEBENARAN: HANYA DATA KONTEKS. Dilarang mengarang angka/fakta di luar konteks.\n\n"
    "FORMAT WAJIB (ikuti persis, gunakan Markdown):\n"
    "#### 📊 RINGKASAN EKSEKUTIF\n"
    "2-3 kalimat: kondisi keseluruhan, X anomali dari Y series, arah dominan, tingkat risiko (RENDAH/SEDANG/TINGGI).\n"
    "#### ⚡ ENERGI\n"
    "Bullet per series: **NAMA**: arah & deviasi % — status (ANOMALI/normal) + implikasi satu klausa. Jika semua normal tulis 'stabil'.\n"
    "#### 🌾 PANGAN\n"
    "Bullet per series dengan format sama.\n"
    "#### 💹 FINANSIAL\n"
    "Bullet per series dengan format sama.\n"
    "#### 🎯 REKOMENDASI\n"
    "2-3 poin bernomor, spesifik, dapat dieksekusi; prioritaskan series ANOMALI.\n\n"
    "ATURAN: maksimal 250 kata; wajib menyebut angka deviasi untuk setiap ANOMALI; tanpa basa-basi."
)

def _f(v):
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "-"

def build_context():
    lines = [f"WAKTU LAPORAN: {datetime.now():%Y-%m-%d %H:%M}"]
    lines.append("DATA KONTEKS (prediksi terbaru per series):")

    if FORECAST_PATH.exists():
        fc = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp", "created_at"])
        latest = fc.sort_values("created_at").groupby("series_id").tail(1)
        for _, r in latest.iterrows():
            lines.append(
                f"- {r['series_id']} [{r['domain']}] terakhir={_f(r['last_value'])} "
                f"prediksi={_f(r['predicted_next_value'])} interval=[{_f(r.get('lower_bound'))} , {_f(r.get('upper_bound'))}] "
                f"signal={r['signal']} deviasi_pola={_f(r.get('deviation_pct'))}% anomali={r['warning']}"
            )

    if METRICS_PATH.exists():
        m = pd.read_csv(METRICS_PATH)
        lines.append("KINERJA MODEL (MAPE% validasi): " +
                     ", ".join(f"{r.series_id}={r.mape_pct}" for r in m.itertuples()))

    if EVAL_PATH.exists():
        e = pd.read_csv(EVAL_PATH)
        if not e.empty and "within_80" in e.columns:
            cov = e.groupby("series_id")["within_80"].apply(lambda s: s.dropna().mean() * 100)
            txt = ", ".join(f"{k}={v:.0f}%" for k, v in cov.items() if pd.notna(v))
            if txt:
                lines.append("COVERAGE LIVE 80%: " + txt)

    text = "\n".join(lines)
    return text, hashlib.md5(text.encode()).hexdigest()

def diagnose_groq():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("❌ GROQ_API_KEY belum ada di file .env", flush=True)
        return
    print(f"🔑 API Key terdeteksi: {key[:8]}...{key[-4:]}", flush=True)
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        if r.status_code == 401:
            print(f"❌ Key TIDAK VALID! {r.text}", flush=True)
            return
        if r.status_code == 200:
            ids = [m.get("id") for m in r.json().get("data", [])]
            print(f"✅ Key VALID! {len(ids)} model tersedia.", flush=True)
            for m in GROQ_MODELS:
                if m in ids:
                    print(f"   ✅ {m}", flush=True)
        else:
            print(f"⚠️ Status {r.status_code}: {r.text[:300]}", flush=True)
    except Exception as e:
        print(f"⚠️ Gagal menghubungi Groq: {e}", flush=True)

def call_groq(context_text: str):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None, "GROQ_API_KEY tidak dikonfigurasi"

    last_err = None
    for model in GROQ_MODELS:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": context_text},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 700,
                },
                timeout=30,
            )
            if r.status_code == 429:
                last_err = "rate limit"
                time.sleep(5)
                continue
            if r.status_code in (400, 404):
                last_err = f"model {model} tidak tersedia"
                continue
            if r.status_code == 401:
                return None, f"API key tidak valid: {r.text[:100]}"
            r.raise_for_status()
            print(f"⚡ Model Groq aktif: {model}", flush=True)
            return r.json()["choices"][0]["message"]["content"].strip(), None
        except Exception as e:
            last_err = str(e)
    return None, last_err or "semua model gagal"

# ---------- Fallback rule-based dengan FORMAT IDENTIK ----------
def rule_based_conclusion():
    if not FORECAST_PATH.exists():
        return "#### 📊 RINGKASAN EKSEKUTIF\nData belum tersedia."

    fc = pd.read_csv(FORECAST_PATH, parse_dates=["created_at"])
    latest = fc.sort_values("created_at").groupby("series_id").tail(1)
    is_anom = latest["warning"].astype(str).str.lower().eq("true")
    n_anom = int(is_anom.sum())
    risk = "TINGGI" if n_anom >= 4 else ("SEDANG" if n_anom >= 2 else "RENDAH")

    def bullets(dom):
        subs = latest[latest["domain"] == dom]
        if subs.empty:
            return ["- stabil."]
        out = []
        for _, r in subs.iterrows():
            if str(r.get("warning")).lower() == "true":
                out.append(
                    f"- **{r['series_id']}**: {r['signal']} dengan deviasi {_f(r.get('deviation_pct'))}% "
                    f"— **ANOMALI**, perlu pemantauan ketat."
                )
            else:
                out.append(f"- **{r['series_id']}**: {str(r['signal']).lower()} dalam batas normal.")
        return out

    anom_names = ", ".join(latest.loc[is_anom, "series_id"]) if n_anom else "-"

    recs = []
    if n_anom:
        recs.append(f"1. Pantau ketat {anom_names} hingga deviasi kembali ke batas normal.")
        recs.append("2. Tinjau eksposur biaya/posisi yang sensitif terhadap series berstatus ANOMALI.")
    else:
        recs.append("1. Tidak ada tindakan khusus; seluruh series dalam pola normal.")
    recs.append(f"{len(recs)+1}. Evaluasi ulang pada siklus berikutnya; validasi model jika coverage live < 70%.")

    return (
        "#### 📊 RINGKASAN EKSEKUTIF\n"
        f"Pasar berada dalam kondisi risiko {risk} dengan {n_anom} anomali dari {len(latest)} series terpantau. "
        + ("Pergerakan dominan berada dalam pola normal." if not n_anom
           else "Fokus diarahkan pada series berstatus ANOMALI.")
        + "\n#### ⚡ ENERGI\n" + "\n".join(bullets("energy"))
        + "\n#### 🌾 PANGAN\n" + "\n".join(bullets("food"))
        + "\n#### 💹 FINANSIAL\n" + "\n".join(bullets("crypto")) + "\n" + "\n".join(bullets("stock"))
        + "\n#### 🎯 REKOMENDASI\n" + "\n".join(recs)
    )

def analyze(force: bool = False):
    text, ctx_hash = build_context()

    if OUTPUT_PATH.exists() and not force:
        old = pd.read_csv(OUTPUT_PATH)
        if not old.empty and str(old.iloc[-1]["context_hash"]) == ctx_hash:
            print("ℹ️ Konteks tidak berubah — kesimpulan sebelumnya tetap dipakai.", flush=True)
            return

    conclusion, err = call_groq(text)
    if conclusion:
        engine = "groq"
        print("✅ LLM sukses via [groq]", flush=True)
    else:
        conclusion = rule_based_conclusion()
        engine = "rule-based"
        print(f"⚠️ Groq gagal: {err}", flush=True)
        print("ℹ️ Fallback ke rule-based (format identik).", flush=True)

    record = pd.DataFrame([{
        "created_at": datetime.now(),
        "engine": engine,
        "context_hash": ctx_hash,
        "conclusion": conclusion,
    }])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        old = pd.read_csv(OUTPUT_PATH, parse_dates=["created_at"])
        combined = pd.concat([old, record], ignore_index=True)
    else:
        combined = record
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"🤖 Kesimpulan dibuat via [{engine}]:", flush=True)
    print(conclusion, flush=True)

if __name__ == "__main__":
    if "--models" in sys.argv or "--diagnose" in sys.argv:
        diagnose_groq()
    else:
        analyze(force=True)