import time

from src.ingest_crypto import main as ingest_crypto
from src.ingest_stock import main as ingest_stock
from src.forecast_latest import forecast
from src.evaluate import evaluate
from src.analyst import analyze

INTERVAL_SECONDS = 3600  # 1 jam

def run():
    while True:
        print("=" * 50, flush=True)
        try:
            ingest_crypto()
            ingest_stock()
            forecast()
            evaluate()
        except Exception as e:
            print("ERROR:", e, flush=True)
        print(f"Menunggu {INTERVAL_SECONDS} detik...", flush=True)
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    run()