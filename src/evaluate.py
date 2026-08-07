import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
FORECAST_PATH = Path("data/forecasts/forecast.csv")
OUTPUT_PATH = Path("data/forecasts/evaluation.csv")

def load_actuals():
    frames = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["timestamp"])[["timestamp", "series_id", "value"]]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def evaluate():
    if not FORECAST_PATH.exists():
        print("❌ Forecast belum ada.", flush=True)
        return

    fc = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp", "created_at"])
    fc = fc.dropna(subset=["predicted_next_value"])

    # Satu evaluasi per (series, timestamp): pakai prediksi terbaru
    fc = fc.sort_values("created_at").drop_duplicates(
        subset=["series_id", "timestamp"], keep="last"
    )

    actuals = load_actuals()
    rows = []

    for (sid, ts), rec in fc.set_index(["series_id", "timestamp"]).iterrows():
        series_actual = actuals[actuals["series_id"] == sid].sort_values("timestamp")
        future = series_actual[series_actual["timestamp"] > ts]
        if future.empty:
            continue  # masih pending: aktual belum terjadi

        actual = float(future.iloc[0]["value"])
        pred = float(rec["predicted_next_value"])
        lower = rec.get("lower_bound")
        upper = rec.get("upper_bound")

        within = np.nan
        if pd.notna(lower) and pd.notna(upper):
            within = bool(lower <= actual <= upper)

        rows.append({
            "series_id": sid,
            "timestamp": ts,
            "predicted": pred,
            "actual": actual,
            "abs_error": abs(actual - pred),
            "ape_pct": abs(actual - pred) / actual * 100 if actual != 0 else np.nan,
            "within_80": within,
        })

    if not rows:
        print("⏳ Belum ada prediksi yang bisa dievaluasi (semua masih pending).", flush=True)
        return

    eval_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUTPUT_PATH, index=False)

    summary = eval_df.groupby("series_id").agg(
        n_eval=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mape_pct=("ape_pct", "mean"),
        coverage_pct=("within_80", lambda s: round(s.dropna().mean() * 100, 1) if s.notna().any() else np.nan),
    ).round(2)

    print("📊 TRACK RECORD LIVE (prediksi vs aktual):", flush=True)
    print(summary.to_string(), flush=True)
    print(f"💾 Disimpan ke {OUTPUT_PATH}", flush=True)

if __name__ == "__main__":
    evaluate()