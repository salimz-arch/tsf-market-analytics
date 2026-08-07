import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

def generate_energy(days: int = 90) -> pd.DataFrame:
    """Simulasi permintaan listrik per jam (MW) dengan pola harian + puncak malam + weekend drop."""
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=24 * days, freq="h")
    hour = idx.hour
    dow = idx.dayofweek

    base = 15000
    daily = 1500 * np.sin(2 * np.pi * (hour - 9) / 24)
    evening_peak = 900 * np.exp(-((hour - 19) ** 2) / 8)
    weekend = -700 * (dow >= 5)
    trend = np.linspace(0, 350, len(idx))
    noise = np.random.normal(0, 220, len(idx))

    value = base + daily + evening_peak + weekend + trend + noise

    return pd.DataFrame({
        "timestamp": idx,
        "series_id": "LOAD_JKT",
        "domain": "energy",
        "value": value.round(1),
        "volume": 0,
    })

def generate_food(days: int = 180, series_id: str = "RICE_JKT",
                  base: float = 13500, noise_scale: float = 110, spike: float = 700) -> pd.DataFrame:
    """Simulasi harga bahan pokok harian (Rp/kg) dengan trend + musiman + spike gangguan."""
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=days, freq="D")
    t = np.arange(len(idx))

    trend = 0.9 * t
    weekly = 120 * np.sin(2 * np.pi * t / 7)
    monthly = 350 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, noise_scale, len(idx))

    shock = np.zeros(len(idx))
    shock[110:122] = spike  # simulasi gangguan pasokan / lonjakan hari besar

    value = base + trend + weekly + monthly + noise + shock

    return pd.DataFrame({
        "timestamp": idx,
        "series_id": series_id,
        "domain": "food",
        "value": value.round(0),
        "volume": 0,
    })

def main():
    np.random.seed(42)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    energy = generate_energy()
    energy.to_csv(RAW_DIR / "energy_load.csv", index=False)
    print(f"✅ Data energi (simulasi): {len(energy)} baris", flush=True)

    rice = generate_food(series_id="RICE_JKT", base=13500)
    rice.to_csv(RAW_DIR / "food_rice.csv", index=False)
    print(f"✅ Data beras (simulasi): {len(rice)} baris", flush=True)

    chili = generate_food(series_id="CHILI_JKT", base=45000, noise_scale=900, spike=4000)
    chili.to_csv(RAW_DIR / "food_chili.csv", index=False)
    print(f"✅ Data cabai (simulasi): {len(chili)} baris", flush=True)

if __name__ == "__main__":
    main()