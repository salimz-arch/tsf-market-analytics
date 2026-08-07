import pandas as pd
from pathlib import Path

from src.features import make_features, FEATURE_COLUMNS

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/dataset.csv")
HORIZON = 1

def build():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Belum ada data di data/raw. Jalankan ingestion dulu.")

    frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    combined = pd.concat(frames, ignore_index=True)

    combined = make_features(combined, target_col="value")
    combined["target"] = combined.groupby("series_id")["value"].shift(-HORIZON)
    combined = combined.dropna(subset=FEATURE_COLUMNS + ["target"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Dataset gabungan: {len(combined)} baris dari {len(files)} file.", flush=True)
    print(combined["series_id"].value_counts().to_string(), flush=True)

if __name__ == "__main__":
    build()