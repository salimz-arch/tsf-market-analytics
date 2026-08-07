from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
import joblib

from src.features import make_features, FEATURE_COLUMNS

app = FastAPI(title="Unified Time Series Forecast API", version="0.2.0")

RAW_DIR = Path("data/raw")
MODEL_DIR = Path("models")

def safe_model_name(series_id: str) -> str:
    return series_id.replace("/", "_").replace(" ", "_").replace(".", "_")

@app.get("/")
def health():
    return {"status": "ok", "message": "API forecasting aktif"}

@app.get("/series")
def list_series():
    series = []
    for raw_file in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(raw_file)
        for sid, dom in df[["series_id", "domain"]].drop_duplicates().values:
            series.append({"series_id": sid, "domain": dom})
    return {"series": series}

@app.get("/predict")
def predict(series_id: str = "BTC_USD"):
    model_path = MODEL_DIR / f"{safe_model_name(series_id)}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model untuk {series_id} belum ada. Jalankan training.")

    for raw_file in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(raw_file, parse_dates=["timestamp"])
        if series_id not in df["series_id"].unique():
            continue

        feats = make_features(df, "value")
        last_row = feats[feats["series_id"] == series_id].iloc[[-1]]

        model = joblib.load(model_path)
        prediction = float(model.predict(last_row[FEATURE_COLUMNS])[0])

        return {
            "series_id": series_id,
            "domain": str(last_row["domain"].iloc[0]) if "domain" in last_row.columns else "unknown",
            "timestamp": str(last_row["timestamp"].iloc[0]),
            "last_value": float(last_row["value"].iloc[0]),
            "predicted_next_value": prediction,
        }

    raise HTTPException(status_code=404, detail=f"Data untuk {series_id} tidak ditemukan.")