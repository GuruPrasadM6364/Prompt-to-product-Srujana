"""
Circular Grid – central configuration.
All tuneable constants live here. Edit this file only.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
UPLOAD_DIR = Path(
    r"C:\Users\Guru Prasad M\.gemini\antigravity\brain"
    r"\430a98f7-876c-4d4a-9647-91604f46ab99\.user_uploaded"
)

# ── Raw file mapping (original xlsx, never edited) ────────────────────────────
RAW_FILES = {
    "buildings":          UPLOAD_DIR / "media_1792821560942_512cf9d6.xlsx",
    "calendar":           UPLOAD_DIR / "media_1792821560942_231837c3.xlsx",
    "daily_summary":      UPLOAD_DIR / "media_1792821560944_41eb6649.xlsx",
    "hourly_streams":     UPLOAD_DIR / "media_1792821560946_8006d6d8.xlsx",
    "manual_logs":        UPLOAD_DIR / "media_1792821560948_725f035c.xlsx",
}

# ── Converted CSV paths (written once by make data) ──────────────────────────
CSV = {k: DATA_DIR / f"{k}.csv" for k in RAW_FILES}
# Waste events may be embedded inside one of the xlsx files; paths set after audit
WASTE_TRAIN_CSV  = DATA_DIR / "waste_events_train.csv"
WASTE_TEST_CSV   = DATA_DIR / "waste_events_test.csv"

# ── Campus / building metadata ────────────────────────────────────────────────
BUILDING_NAMES = {
    "B01": "Academic Block A",
    "B02": "Academic Block B",
    "B03": "Central Library",
    "B04": "Engineering Labs & Innovation Centre",
    "B05": "Boys Hostel",
    "B06": "Girls Hostel",
    "B07": "Food Court",
    "B08": "Admin Block",
}
SOLAR_BUILDINGS    = {"B01", "B04", "B07"}
COMPOSTER_BUILDINGS = {"B05", "B06", "B07"}
ALL_BUILDINGS      = list(BUILDING_NAMES.keys())

DATA_START = "2026-06-20"
DATA_END   = "2026-10-17"
N_DAYS     = 120
N_BUILDINGS = 8

# ── Unit / cost constants (from README_data_guide.md §6) ─────────────────────
KWH_PER_UNIT           = 1.0          # "1 unit = 1 kWh"
LITRES_PER_KL          = 1_000.0
KG_PER_BAG             = 8.0          # ~8 kg per bag
COMPOST_RATIO          = 4.0          # 4 kg wet organic → 1 kg compost
RS_PER_KWH             = 8.0          # ₹ per kWh
KG_CO2_PER_KWH         = 0.70         # kg CO2 per kWh

# ── Classifier settings ────────────────────────────────────────────────────────
WASTE_CLASSES = [
    "food_organic", "plastic", "paper_cardboard",
    "metal", "ewaste", "glass",
]
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.55   # below → manual sort
CLASSIFIER_LATENCY_BUDGET_S     = 2.0    # per-event SLA
RANDOM_SEED                     = 42

# ── Forecasting ───────────────────────────────────────────────────────────────
FORECAST_HORIZONS   = [6, 12]           # hours ahead
FORECAST_TARGETS    = ["electricity_kwh", "water_litres", "footfall"]
SPIKE_PERCENTILE    = 90                # above this = spike

# ── Anomaly / incentive ───────────────────────────────────────────────────────
ANOMALY_Z_THRESHOLD = 3.0
CREDIT_CAP_PER_DAY  = 50               # credits / occupant / day
NIGHT_HOURS         = set(range(0, 6)) | {23}  # midnight-5am + 11pm

# ── Edge / connectivity ───────────────────────────────────────────────────────
STALE_THRESHOLD_HOURS    = 2
OFFLINE_THRESHOLD_HOURS  = 6
OUTAGE_BUILDINGS         = ["B03", "B07"]   # for replay test
OUTAGE_DURATION_HOURS    = 6

# ── Dashboard ─────────────────────────────────────────────────────────────────
APP_TITLE           = "Circular Grid"
PRIMARY_COLOR       = "#1a6e3c"         # single primary, not purple
DASHBOARD_PORT      = 8000
