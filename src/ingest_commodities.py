import time
import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

COMMODITIES = [
    ("SB=F", "SUGAR_GLOBAL", "food"),
    ("ZW=F", "WHEAT_GLOBAL", "food"),
    ("ZC=F", "CORN_GLOBAL", "food"),
    ("NG=F", "NATGAS_GLOBAL", "energy"),
    ("CL=F", "OIL_GLOBAL", "energy"),
]

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, series_id, domain in COMMODITIES:
        try:
            print(f"Mengambil {symbol} ({domain})...", flush=True)
            df = yf.Ticker(symbol).history(period="2y", interval="1d")

            if df.empty:
                print(f"⚠️ {symbol} kosong, dilewati.", flush=True)
                continue

            df = df.reset_index()
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df = df.rename(columns={ts_col: "timestamp", "Close": "value", "Volume": "volume"})
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df["series_id"] = series_id
            df["domain"] = domain

            out = RAW_DIR / f"commodity_{series_id.lower()}.csv"
            df[["timestamp", "series_id", "domain", "value", "volume"]].to_csv(out, index=False)
            print(f"✅ {series_id}: {len(df)} baris data ASLI", flush=True)

            time.sleep(1)  # sopan ke API
        except Exception as e:
            print(f"⚠️ {symbol} gagal: {e}", flush=True)

if __name__ == "__main__":
    main()