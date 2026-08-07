# TSF Analytics — Unified Real-Time Time Series Forecasting

Platform forecasting **multi-domain** yang memprediksi harga kripto, saham, permintaan energi,
dan komoditas pangan **secara real-time** — dilengkapi interval keyakinan 80%, early warning
cerdas, audit akurasi otomatis, analis AI (Groq), dashboard interaktif, REST API, dan
notifikasi Telegram.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-purple) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)

---

## ✨ Fitur Lengkap

### 1. Unified Multi-Domain Pipeline
Satu skema data (`timestamp, series_id, domain, value`) dan satu pipeline untuk semua domain:
kripto, saham, energi, dan pangan. Menambah series baru cukup dengan menaruh CSV di `data/raw/`.

### 2. Probabilistic Forecasting (Interval Keyakinan 80%)
Setiap series dilatih dengan **3 model LightGBM**: titik (P50) + quantile **P10 & P90**.
Dashboard menampilkan pita ketidakpastian; interval lebar = model ragu, interval sempit = model yakin.

### 3. Smart Early Warning (Seasonal-Aware)
Alarm **tidak** menyala hanya karena harga berubah — tetapi hanya ketika prediksi **menyimpang
dari pola musiman normal** (baseline jam-untuk energi, hari-untuk pangan harian, median 28 hari
untuk data harian bertrend). Anti false-alarm: beban listrik turun tiap malam ≠ anomali.

### 4. Sinyal & Top Movers
Sinyal `UP / DOWN / FLAT` per series dengan threshold per domain, plus peringkat pergerakan
terbesar (Top Movers) di dashboard.

### 5. Live Track Record (Self-Auditing)
Sistem mencocokkan setiap prediksi dengan aktual yang terjadi kemudian:
**MAE / MAPE live** dan **coverage interval** (ideal ≈ 80%). Model yang memburuk akan terlihat.

### 6. AI Analyst (Groq Cloud — Llama-3.3-70B)
Laporan pasar gaya **riset institusional** (Ringkasan Eksekutif → Energi → Pangan → Finansial →
Rekomendasi) dengan format terkunci agar **konsisten**, temperature 0.1, hash-cached (hemat kuota),
dan **fallback rule-based deterministik** bila API down.

### 7. Dashboard Interaktif 5 Halaman (Streamlit)
Overview (KPI, Market Pulse gradient, donut domain, peringatan, aktivitas) • Forecast & Interval •
Kinerja Model • Laporan AI • Data Historis. Tema dark analytics profesional, CSS terpisah.

### 8. REST API (FastAPI)
`GET /` health • `GET /series` daftar series • `GET /predict?series_id=BTC_USD` prediksi terbaru.

### 9. Notifikasi Telegram
Laporan AI + peringatan anomali terkirim otomatis ke Telegram tiap siklus.

---

## 🌐 Sumber Data & Mode Real-Time

| Domain | Series | Sumber | Frekuensi | Mode Update |
|---|---|---|---|---|
| Kripto | `BTC_USD` | **Yahoo Finance** (`BTC-USD`) via `yfinance` | 1 jam | Real-time: loop tiap 1 jam |
| Saham | `BBCA_JK` | **Yahoo Finance IDX** (`BBCA.JK`) via `yfinance` | 1 jam | Loop tiap 1 jam (jam bursa) |
| Energi | `NATGAS_GLOBAL` (`NG=F`), `OIL_GLOBAL` (`CL=F`) | **Yahoo Finance futures** — Henry Hub & WTI | harian | Dicek tiap jam |
| Energi (lokal) | `LOAD_JKT` | Simulasi beban listrik Jakarta (siap diganti data **PLN/ESDM/SCADA**) | 1 jam | — |
| Pangan | `SUGAR_GLOBAL` (`SB=F`), `WHEAT_GLOBAL` (`ZW=F`), `CORN_GLOBAL` (`ZC=F`) | **Yahoo Finance commodity futures** — acuan harga pangan global | harian | Dicek tiap jam |

> **Alur real-time:** `src/realtime_loop.py` menjalankan setiap 3600 detik:
> ingestion → feature engineering → prediksi + interval → smart warning → evaluasi live →
> laporan AI → kirim Telegram.

---
## 🏗️ Arsitektur

```
Yahoo Finance (yfinance: kripto/saham/futures)
        │  (REST, interval 1h/1d)
        ▼
Ingestion & validasi  →  data/raw/*.csv
        ▼
Feature engineering (lag, rolling, kalender — leak-proof via shift(1))
        ▼
LightGBM per-series: P10 / P50 / P90  →  models/*.joblib
        ▼
Smart warning (seasonal baseline) + sinyal UP/DOWN/FLAT
        ▼
Live evaluation (prediksi vs aktual)  →  data/forecasts/evaluation.csv
        ▼
Groq AI analyst (fallback rule-based)  →  conclusions.csv
        ▼
Streamlit dashboard │ FastAPI │ Telegram
```

## Struktur Proyek

- `dashboard/`
  - `app.py` : Streamlit dashboard untuk visualisasi dan laporan KPI
  - `style.css` : gaya visual dashboard
- `data/`
  - `raw/` : file data historis input untuk setiap seri waktu
  - `processed/` : dataset olahan yang siap dipakai untuk training
  - `forecasts/` : hasil prediksi, evaluasi, dan kesimpulan
- `models/`
  - Model siap pakai (`*.joblib`)
  - `metrics.csv` : metrik kinerja model
- `src/`
  - `api.py` : FastAPI service untuk forecasting
  - `features.py` : pembuatan fitur untuk model
  - `train.py` : pipeline pelatihan model
  - `evaluate.py` : evaluasi hasil model
  - `forecast_latest.py` : pembuatan prediksi terbaru
  - `ingest_*.py` : skrip ingest untuk berbagai sumber data
  - `build_dataset.py` : pembuatan dataset terintegrasi
  - `realtime_loop.py` : loop real-time untuk pembaruan data/forecast
  - `notifier.py` : notifikasi/peringatan berbasis hasil forecast

## Prasyarat

- Python 3.9+ (direkomendasikan)
- Windows, macOS, atau Linux
- `pip` untuk instalasi paket

## 🚀 Instalasi & Menjalankan

```bash
git clone <tsf-market-analytics> && cd time-series-forecasting
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Secret API 
echo GROQ_API_KEY=gsk_xxx > .env
echo TELEGRAM_BOT_TOKEN=xxx >> .env
echo TELEGRAM_CHAT_ID=xxx >> .env

# Pipeline pertama
python -m src.ingest_crypto
python -m src.ingest_stock
python -m src.ingest_commodities
python -m src.build_dataset
python -m src.train
python -m src.forecast_latest
python -m src.evaluate
python -m src.analyst

# Jalankan layanan
streamlit run dashboard/app.py           # dashboard
python -m uvicorn src.api:app --reload   # REST API
python -m src.realtime_loop              # mode real-time 24/7
```

## Alur Penggunaan

1. Siapkan data mentah di `data/raw/` dalam format CSV.
2. Jalankan pipeline untuk membuat dataset dan melatih model jika perlu.
3. Gunakan dashboard untuk memonitor hasil forecast dan metrik kinerja.
4. Akses API untuk menarik prediksi programatik.


## Catatan Penting

- Model disimpan dalam format `joblib` di folder `models/`
- Prediksi API membaca file data mentah dan memproses fitur sebelum menjalankan model
- Dashboard menggunakan file forecast/evaluasi di `data/forecasts/`

## Pengembangan dan Ekstensi

Untuk memperbarui pipeline atau menambahkan seri baru:

- `src/build_dataset.py` : membangun dataset historis
- `src/train.py` : melatih model baru
- `src/evaluate.py` : menghitung metrik model
- `src/forecast_latest.py` : membuat prediksi terbaru
- `src/realtime_loop.py` : jalankan loop bila ingin memproses data secara berkelanjutan

## Pemecahan Masalah

- Jika dashboard tidak menampilkan data: pastikan `data/forecasts/forecast.csv` dan `data/raw/` berisi file CSV yang valid.
- Jika API `predict` gagal karena model tidak ditemukan: pastikan file model `models/<series_id>.joblib` tersedia.
- Jika instalasi paket gagal: periksa versi Python dan jalankan kembali `pip install -r requirements.txt`.

## Lisensi


