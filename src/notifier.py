import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONC_PATH = Path("data/forecasts/conclusions.csv")

def send_telegram(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ℹ️ Telegram belum dikonfigurasi. Isi TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID di .env", flush=True)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        r.raise_for_status()
        print("📬 Terkirim ke Telegram.", flush=True)
        return True
    except Exception as e:
        print(f"⚠️ Gagal kirim Telegram: {e}", flush=True)
        return False

def send_latest_report() -> bool:
    """Kirim laporan AI terbaru ke Telegram (format plain-text agar aman)."""
    if not CONC_PATH.exists():
        print("ℹ️ Laporan belum ada. Jalankan python -m src.analyst dulu.", flush=True)
        return False

    df = pd.read_csv(CONC_PATH, parse_dates=["created_at"])
    if df.empty:
        return False

    last = df.iloc[-1]

    # Bersihkan markdown dashboard agar rapi di Telegram
    body = str(last["conclusion"]).replace("#### ", "").replace("**", "")

    header = (
        "📡 LAPORAN PASAR OTOMATIS\n"
        f"{last['created_at']:%Y-%m-%d %H:%M} • engine: {last['engine']}\n"
        "──────────────────────────────\n"
    )
    return send_telegram(header + body)

if __name__ == "__main__":
    if not send_latest_report():
        send_telegram("✅ Test alert dari Unified Time Series Forecasting!")