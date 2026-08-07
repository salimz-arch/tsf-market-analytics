import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

for f in sorted(RAW_DIR.glob("*.csv")):
    df = pd.read_csv(f)
    n0 = len(df)

    # buang baris kosong & baris geser (series_id berupa angka)
    df = df.dropna(subset=["series_id", "value"])
    df["series_id"] = df["series_id"].astype(str).str.strip()
    df = df[~df["series_id"].str.match(r"^\d+(\.\d+)?$")]
    df = df[df["series_id"] != ""]

    if len(df) != n0:
        df.to_csv(f, index=False)
        print(f"🧹 {f.name}: {n0 - len(df)} baris rusak dibuang")
    else:
        print(f"✅ {f.name}: bersih")