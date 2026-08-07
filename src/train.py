import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.features import FEATURE_COLUMNS

DATASET_PATH = Path("data/processed/dataset.csv")
MODEL_DIR = Path("models")

try:
    import lightgbm as lgb
    USE_LGBM = True
except Exception:
    USE_LGBM = False

def safe_model_name(series_id: str) -> str:
    return series_id.replace("/", "_").replace(" ", "_").replace(".", "_")

def train_point_model(X_train, y_train):
    if USE_LGBM:
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_quantile_model(X_train, y_train, alpha):
    if USE_LGBM:
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                                  n_estimators=300, learning_rate=0.05,
                                  random_state=42, verbose=-1)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                          n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def train():
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Dataset belum ada. Jalankan: python -m src.build_dataset")

    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    MODEL_DIR.mkdir(exist_ok=True)

    metrics_rows = []
    print(f"Melatih model untuk {df['series_id'].nunique()} series...", flush=True)

    for series_id, group in df.groupby("series_id"):
        group = group.sort_values("timestamp")
        X = group[FEATURE_COLUMNS]
        y = group["target"]

        split_index = int(len(group) * 0.8)
        if split_index < 50 or (len(group) - split_index) < 10:
            print(f"⚠️ {series_id}: data terlalu sedikit, dilewati.", flush=True)
            continue

        X_train, X_valid = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_valid = y.iloc[:split_index], y.iloc[split_index:]

        # 1) Model utama (P50 / titik)
        model = train_point_model(X_train, y_train)
        predictions = model.predict(X_valid)

        # 2) Model quantile (P10 & P90) untuk confidence interval
        base_name = safe_model_name(series_id)
        joblib.dump(model, MODEL_DIR / f"{base_name}.joblib")
        for alpha, suffix in [(0.1, "q10"), (0.9, "q90")]:
            qmodel = train_quantile_model(X_train, y_train, alpha)
            joblib.dump(qmodel, MODEL_DIR / f"{base_name}_{suffix}.joblib")

        mae = mean_absolute_error(y_valid, predictions)
        rmse = np.sqrt(mean_squared_error(y_valid, predictions))
        mape = float(np.mean(np.abs((y_valid.values - predictions) / y_valid.values)) * 100)

        metrics_rows.append({
            "series_id": series_id,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape_pct": round(mape, 2),
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"✅ {series_id}: MAE={mae:.2f} | MAPE={mape:.2f}% (+interval P10/P90)", flush=True)

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(MODEL_DIR / "metrics.csv", index=False)
        print("📊 Metrik disimpan ke models/metrics.csv", flush=True)

if __name__ == "__main__":
    train()