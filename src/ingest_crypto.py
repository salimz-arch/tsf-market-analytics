import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_crypto_yfinance(symbol: str = "BTC-USD", interval: str = "1h", period: str = "60d") -> pd.DataFrame:
    """
    Mengambil data kripto dari Yahoo Finance.
    Simbol kripto di Yahoo Finance biasanya berakhiran -USD, contoh: BTC-USD, ETH-USD.
    """
    print(f"Mengambil data {symbol} dari Yahoo Finance...")
    ticker = yf.Ticker(symbol)
    
    # Ambil data historis
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"Tidak ada data yang berhasil diambil untuk {symbol}.")
        
    # Reset index agar timestamp menjadi kolom biasa
    df = df.reset_index()
    
    # Yahoo Finance kadang menamai kolom waktu sebagai 'Datetime' atau 'Date'
    timestamp_col = "Datetime" if "Datetime" in df.columns else "Date"
    
    # Rename kolom agar sesuai dengan skema Unified Project kita
    df = df.rename(columns={
        timestamp_col: "timestamp",
        "Close": "value",
        "Volume": "volume"
    })
    
    # Hapus informasi timezone agar konsisten (naive datetime)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    
    # Tambahkan kolom identitas untuk pipeline multi-domain
    df["series_id"] = symbol.replace("-", "_") # BTC-USD menjadi BTC_USD
    df["domain"] = "crypto"
    
    # Ambil hanya kolom yang kita butuhkan
    return df[["timestamp", "series_id", "domain", "value", "volume"]]

def main():
    # Kita ambil data BTC-USD, interval 1 jam, periode 60 hari ke belakang
    # (Catatan: yfinance membatasi data intraday 1h maksimal 730 hari)
    df = fetch_crypto_yfinance(symbol="BTC-USD", interval="1h", period="60d")
    
    output_path = Path("data/raw/btc_1h.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"✅ Data berhasil disimpan ke {output_path}")
    print(f"📊 Jumlah baris: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    main()