import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_stock_yfinance(symbol: str = "BBCA.JK", interval: str = "1h", period: str = "60d") -> pd.DataFrame:
    print(f"Mengambil data {symbol} dari Yahoo Finance...", flush=True)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"Tidak ada data untuk {symbol}.")

    df = df.reset_index()
    timestamp_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={timestamp_col: "timestamp", "Close": "value", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["series_id"] = symbol.replace(".JK", "_JK")
    df["domain"] = "stock"

    return df[["timestamp", "series_id", "domain", "value", "volume"]]

def main():
    df = fetch_stock_yfinance(symbol="BBCA.JK", interval="1h", period="60d")
    output_path = Path("data/raw/stock_1h.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Data saham disimpan ke {output_path}", flush=True)
    print(f"📊 Jumlah baris: {len(df)}", flush=True)

if __name__ == "__main__":
    main()