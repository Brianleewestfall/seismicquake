# 🌍 SeismicQuake — TeslaQuake Edition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/TeslaQuake-Integrated-C49A3C.svg" alt="TeslaQuake">
  <img src="https://img.shields.io/badge/Supabase-Connected-3ECF8E.svg" alt="Supabase">
</p>

AI-powered earthquake detection with **TeslaQuake dual-frequency analysis** — detects earthquakes, classifies seismic waves, predicts magnitudes, and correlates electromagnetic precursors.

**Forked from [JustineBijuPaul/seismicquake](https://github.com/JustineBijuPaul/seismicquake)** and customized for [TeslaQuake](https://teslaquake.com) earthquake prediction research.

---

## TeslaQuake Additions

| File | Purpose |
|------|---------|
| `supabase_bridge.py` | Push AI results to TeslaQuake Supabase database |
| `teslaquake_pipeline.py` | End-to-end: ObsPy → AI → Frequency Analysis → Supabase |
| `historical_validation.py` | Backtest AI accuracy against 14K+ historical earthquakes |

### Full Pipeline

```
ObsPy downloads .mseed waveform
        ↓
SeismicQuake AI (3 models)
  ├── Earthquake Detector (96.8% accuracy)
  ├── Wave Classifier (99.7% — P/S/Surface)
  └── Magnitude Predictor (MAE: 0.37)
        ↓
TeslaQuake FFT Frequency Analysis
  ├── Schumann Resonance 7.83 Hz (SR₁)
  ├── Tesla Telluric 11.78 Hz
  └── 5 additional harmonics
        ↓
Supabase Push
  ├── seismicquake_results (full AI output)
  └── anomaly_detections (flagged events)
        ↓
TeslaQuake Dashboard (auto-displays)
```

---

## AI Models (Pre-trained, included)

| Model | File | Accuracy | Size |
|-------|------|----------|------|
| Earthquake Detector | `earthquake_detector_best.h5` | 96.81% (AUC 99.59%) | 2.1 MB |
| Wave Classifier | `wave_classifier_best.h5` | 99.69% | 2.1 MB |
| Magnitude Predictor | `magnitude_predictor_best.h5` | MAE 0.374 (93% within ±1.0) | 3.9 MB |

Trained on **STEAD (Stanford Earthquake Dataset)** — 1.79M labeled waveform segments.

---

## Quick Start

```bash
git clone https://github.com/Brianleewestfall/seismicquake.git
cd seismicquake

python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install tensorflow numpy pandas h5py obspy scipy matplotlib
```

### Set Supabase credentials (optional)

```bash
# Windows PowerShell
$env:TESLAQUAKE_SUPABASE_URL = "https://your-project.supabase.co"
$env:TESLAQUAKE_SUPABASE_KEY = "your-service-role-key"

# Linux/Mac
export TESLAQUAKE_SUPABASE_URL="https://your-project.supabase.co"
export TESLAQUAKE_SUPABASE_KEY="your-service-role-key"
```

---

## Usage

### 1. Analyze a waveform (standalone)

```bash
python seismic_analyzer.py analyze earthquake.mseed --visualize
```

### 2. Full TeslaQuake Pipeline

```bash
# Analyze an ObsPy download folder
python teslaquake_pipeline.py ./obspy_downloads/2025-12-08T14-15-10Z_IU.ANMO

# Batch all downloads
python teslaquake_pipeline.py ./obspy_downloads --batch

# Local only (no Supabase push)
python teslaquake_pipeline.py ./obspy_downloads/folder --no-push
```

### 3. Historical Validation

```bash
# Full accuracy report (instant — queries Supabase)
python historical_validation.py --report

# Score existing AI results vs USGS
python historical_validation.py --score-existing

# TeslaQuake prediction accuracy
python historical_validation.py --score-predictions

# Frequency → earthquake correlation
python historical_validation.py --frequency-correlation

# Backtest: download + analyze + score (slow, ~3 min/event)
python historical_validation.py --backtest --min-mag 6.0 --max-events 10
```

### 4. Python API

```python
from teslaquake_pipeline import TeslaQuakePipeline
from historical_validation import ValidationEngine

# Run pipeline
pipeline = TeslaQuakePipeline()
result = pipeline.analyze_folder("./obspy_downloads/2025-12-08_IU.ANMO")

# Validate accuracy
engine = ValidationEngine()
report = engine.generate_report()
```

---

## Supabase Tables

| Table | Purpose |
|-------|---------|
| `seismicquake_results` | Full AI analysis output (detection + waves + magnitude + frequency) |
| `anomaly_detections` | Flagged events (high-confidence detections, frequency anomalies) |
| `earthquakes` | 14K+ USGS events (ground truth for validation) |
| `predictions` | 163 TeslaQuake predictions (47 linked to actual quakes) |
| `welford_baselines` | Running frequency statistics |

---

## File Structure

```
seismicquake/
├── earthquake_ai_models/           # Pre-trained AI models (included)
│   ├── earthquake_detector_best.h5
│   ├── wave_classifier_best.h5
│   └── magnitude_predictor_best.h5
├── seismic_analyzer.py             # Core AI engine (original)
├── main.py                         # Legacy entry point
├── train_earthquake_ai.py          # Model training script
├── supabase_bridge.py              # ⚡ TeslaQuake Supabase integration
├── teslaquake_pipeline.py          # ⚡ ObsPy → AI → Supabase pipeline
├── historical_validation.py        # ⚡ Accuracy backtesting engine
└── README.md
```

---

## References

- [ObsPy](https://docs.obspy.org/) — Seismic data processing
- [STEAD Dataset](https://github.com/smousavi05/STEAD) — Training data
- [TeslaQuake](https://teslaquake.com) — Earthquake prediction research
- [simple-obspy MCP](https://github.com/Brianleewestfall/simple-obspy) — Claude Desktop integration

## Acknowledgments

Original SeismicQuake by **Justine Biju Paul**.
TeslaQuake integration by **Brian Lee Westfall** — AI Vision Designs, Fort Worth, TX.

---

**© 2025-2026 Brian Lee Westfall. TeslaQuake additions are proprietary.**
