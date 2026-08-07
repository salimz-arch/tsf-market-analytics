import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from src.features import make_features, FEATURE_COLUMNS

RAW_DIR = Path("data/raw")
MODEL_DIR = Path("models")
OUTPUT_PATH = Path("data/forecasts/forecast.csv")

SIGNAL_THRESHOLD = {"crypto": 0.3, "stock": 0.5, "energy": 2.0, "food": 0.5}
DEVIATION_THRESHOLD = {"energy": 3.0, "food": 3.0}  # % penyimpangan dari pola normal

def safe_model_name(series_id: str) -> str:
    return series_id.replace("/", "_").replace(" ", "_").replace(".", "_")

def seasonal_baseline(df: pd.DataFrame, domain: str, target_ts):
    """Rata-rata historis pada jam yang sama (energi) / hari yang sama (pangan)."""
    if domain == "energy":
        hist = df.copy()
        hist["key"] = hist["timestamp"].dt.hour
        return hist.groupby("key")["value"].mean().get(target_ts.hour)
    if domain == "food":
        hist = df.copy()
        hist["key"] = hist["timestamp"].dt.dayofweek
        return hist.groupby("key")["value"].mean().get(target_ts.dayofweek)
    return None

def forecast():
    records = []

    for raw_file in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(raw_file, parse_dates=["timestamp"])
        feats = make_features(df, target_col="value")
        step = df["timestamp"].diff().dropna().median()

        for series_id, group in feats.groupby("series_id"):
            base_name = safe_model_name(series_id)
            model_path = MODEL_DIR / f"{base_name}.joblib"
            if not model_path.exists():
                print(f"⚠️ Model untuk {series_id} belum ada, dilewati.", flush=True)
                continue

            last_row = group.iloc[[-1]]
            model = joblib.load(model_path)
            prediction = float(model.predict(last_row[FEATURE_COLUMNS])[0])
            last_value = float(last_row["value"].iloc[0])
            domain = str(last_row["domain"].iloc[0]) if "domain" in last_row.columns else "unknown"

            lower = upper = None
            q10_path = MODEL_DIR / f"{base_name}_q10.joblib"
            q90_path = MODEL_DIR / f"{base_name}_q90.joblib"
            if q10_path.exists() and q90_path.exists():
                lower = float(joblib.load(q10_path).predict(last_row[FEATURE_COLUMNS])[0])
                upper = float(joblib.load(q90_path).predict(last_row[FEATURE_COLUMNS])[0])

            change_pct = (prediction - last_value) / last_value * 100 if last_value != 0 else 0.0
            thr = SIGNAL_THRESHOLD.get(domain, 0.5)
            signal = "UP" if change_pct > thr else ("DOWN" if change_pct < -thr else "FLAT")

            # SMART WARNING: bandingkan prediksi dengan pola musiman normal
            baseline = deviation_pct = None
            warning = False
            if domain in DEVIATION_THRESHOLD and step is not None:
                target_ts = last_row["timestamp"].iloc[0] + step
                baseline = seasonal_baseline(df, domain, target_ts)
                if baseline is not None and baseline == baseline and baseline != 0:
                    deviation_pct = round((prediction - baseline) / baseline * 100, 3)
                    warning = abs(deviation_pct) > DEVIATION_THRESHOLD[domain]

            records.append({
                "created_at": datetime.now(),
                "series_id": series_id,
                "domain": domain,
                "timestamp": last_row["timestamp"].iloc[0],
                "last_value": last_value,
                "predicted_next_value": prediction,
                "lower_bound": lower,
                "upper_bound": upper,
                "change_pct": round(change_pct, 3),
                "signal": signal,
                "baseline": round(float(baseline), 2) if baseline is not None and baseline == baseline else None,
                "deviation_pct": deviation_pct,
                "warning": warning,
            })

    if not records:
        print("❌ Tidak ada prediksi yang bisa dibuat.", flush=True)
        return

    record_df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        old = pd.read_csv(OUTPUT_PATH, parse_dates=["created_at", "timestamp"])
        for col in ["change_pct", "signal", "warning", "lower_bound", "upper_bound", "baseline", "deviation_pct"]:
            if col not in old.columns:
                old[col] = None
        combined = pd.concat([old, record_df], ignore_index=True)
    else:
        combined = record_df

    combined.to_csv(OUTPUT_PATH, index=False)

    # ---------- Kirim alert anomali ke Telegram ----------
    from src.notifier import send_telegram

    alerts = [r for r in records if r["warning"]]
    if alerts:
        lines = ["🚨 *PERINGATAN DINI ANOMALI* 🚨"]
        for r in alerts:
            lines.append(
                f"• *{r['series_id']}* ({r['domain']}): "
                f"prediksi {r['predicted_next_value']:,.0f} "
                f"({r['deviation_pct']:+.2f}% vs pola normal)"
            )
        send_telegram("\n".join(lines))

    print("✅ Forecast berhasil dibuat:", flush=True)
    print(record_df[["series_id", "predicted_next_value", "signal", "baseline", "deviation_pct", "warning"]].to_string(index=False), flush=True)

if __name__ == "__main__":
    forecast()