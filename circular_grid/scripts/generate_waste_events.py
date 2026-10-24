"""
generate_waste_events.py
========================
Generates synthetic waste_events_train.csv (6 314 rows) and
waste_events_test.csv (1 601 rows) in the data/ directory.

Class distributions (train):
  food_organic   : 1 852
  plastic        : 1 722
  paper_cardboard: 1 461
  metal          :   608
  ewaste         :   351
  glass          :   320

Sensor distributions (realistic):
  food_organic   – high moisture, medium-high weight, low metal, low transparency, soft
  plastic        – low moisture, medium weight, low metal, HIGH transparency, medium hardness
  metal          – low moisture, high weight, HIGH metal_signal, medium transparency, HIGH hardness
  ewaste         – low moisture, high weight, HIGH metal_signal, medium transparency, HIGH hardness
  glass          – zero moisture, heavy, zero metal, HIGH transparency, HIGH hardness
  paper_cardboard– medium moisture, low weight, zero metal, low transparency, medium hardness

~2 % NaN injection per sensor column.

Usage:  python scripts/generate_waste_events.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Class counts ───────────────────────────────────────────────────────────────
TRAIN_COUNTS = {
    "food_organic":    1852,
    "plastic":         1722,
    "paper_cardboard": 1461,
    "metal":            608,
    "ewaste":           351,
    "glass":            320,
}
TRAIN_TOTAL = sum(TRAIN_COUNTS.values())   # 6 314
TEST_TOTAL  = 1601

BUILDINGS = [f"B{i:02d}" for i in range(1, 9)]
DATE_START = pd.Timestamp("2026-06-20")
DATE_END   = pd.Timestamp("2026-10-17")
SPAN_SECS  = int((DATE_END - DATE_START).total_seconds())

NAN_RATE = 0.02  # ~2% missing per sensor column

# ── Per-class sensor distributions  (mean, std, clip_lo, clip_hi) ────────────
# weight_g:         grams deposited per event
# moisture_pct:     0-100
# metal_signal:     0-1 normalized
# transparency:     0-1 normalized
# acoustic_hardness:0-1 normalized

DIST = {
    #              weight_g              moisture_pct        metal_signal         transparency         acoustic_hardness
    "food_organic":    dict(
        weight_g         = (420, 150, 50, 1200),
        moisture_pct     = (72,  12,  30,  99),
        metal_signal     = (0.05, 0.03, 0, 0.3),
        transparency     = (0.05, 0.04, 0, 0.25),
        acoustic_hardness= (0.15, 0.08, 0, 0.5),
    ),
    "plastic": dict(
        weight_g         = (180, 80,  20, 800),
        moisture_pct     = (8,   5,   0,  30),
        metal_signal     = (0.06, 0.04, 0, 0.25),
        transparency     = (0.72, 0.15, 0.3, 1.0),
        acoustic_hardness= (0.40, 0.12, 0.1, 0.75),
    ),
    "paper_cardboard": dict(
        weight_g         = (120, 60,  10, 600),
        moisture_pct     = (28,  10,  5,  65),
        metal_signal     = (0.04, 0.03, 0, 0.15),
        transparency     = (0.10, 0.06, 0, 0.35),
        acoustic_hardness= (0.28, 0.10, 0.05, 0.60),
    ),
    "metal": dict(
        weight_g         = (560, 200, 80, 2000),
        moisture_pct     = (5,   4,   0,  20),
        metal_signal     = (0.88, 0.08, 0.55, 1.0),
        transparency     = (0.12, 0.08, 0, 0.4),
        acoustic_hardness= (0.88, 0.07, 0.6, 1.0),
    ),
    "ewaste": dict(
        weight_g         = (480, 220, 50, 2500),
        moisture_pct     = (4,   3,   0,  15),
        metal_signal     = (0.80, 0.10, 0.45, 1.0),
        transparency     = (0.18, 0.10, 0, 0.50),
        acoustic_hardness= (0.82, 0.08, 0.55, 1.0),
    ),
    "glass": dict(
        weight_g         = (640, 180, 80, 2200),
        moisture_pct     = (2,   2,   0,  10),
        metal_signal     = (0.03, 0.02, 0, 0.12),
        transparency     = (0.85, 0.10, 0.5, 1.0),
        acoustic_hardness= (0.92, 0.05, 0.75, 1.0),
    ),
}


def _sample_col(n: int, mean: float, std: float, lo: float, hi: float) -> np.ndarray:
    """Draw n values from a truncated normal distribution."""
    vals = rng.normal(loc=mean, scale=std, size=n)
    return np.clip(vals, lo, hi)


def _inject_nans(arr: np.ndarray, rate: float = NAN_RATE) -> np.ndarray:
    """Replace ~rate fraction of values with NaN."""
    arr = arr.astype(float)
    mask = rng.random(len(arr)) < rate
    arr[mask] = np.nan
    return arr


def _make_timestamps(n: int) -> pd.DatetimeIndex:
    offsets = rng.integers(0, SPAN_SECS, size=n)
    ts = [DATE_START + pd.Timedelta(seconds=int(s)) for s in offsets]
    return pd.DatetimeIndex(ts).floor("s")


def _make_rows_for_class(cls: str, n: int) -> pd.DataFrame:
    d = DIST[cls]
    df = pd.DataFrame({
        "weight_g"          : _sample_col(n, *d["weight_g"]),
        "moisture_pct"      : _sample_col(n, *d["moisture_pct"]),
        "metal_signal"      : _sample_col(n, *d["metal_signal"]),
        "transparency"      : _sample_col(n, *d["transparency"]),
        "acoustic_hardness" : _sample_col(n, *d["acoustic_hardness"]),
    })
    df["category"] = cls
    return df


def _generate(counts: dict, include_label: bool, id_prefix: str) -> pd.DataFrame:
    """Build the full dataset from class counts dict.  counts=None for unlabelled test."""
    if include_label:
        parts = [_make_rows_for_class(cls, n) for cls, n in counts.items()]
        df = pd.concat(parts, ignore_index=True)
        # shuffle
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        # For test set: sample classes proportionally from train distribution
        total = TEST_TOTAL
        probs = np.array(list(TRAIN_COUNTS.values()), dtype=float)
        probs /= probs.sum()
        cls_list = list(TRAIN_COUNTS.keys())
        test_counts_arr = rng.multinomial(total, probs)
        test_counts = {cls_list[i]: int(test_counts_arr[i]) for i in range(len(cls_list))}
        parts = [_make_rows_for_class(cls, n) for cls, n in test_counts.items()]
        df = pd.concat(parts, ignore_index=True)
        df = df.sample(frac=1, random_state=RANDOM_SEED + 1).reset_index(drop=True)
        df.drop(columns=["category"], inplace=True)

    n = len(df)
    # Assign timestamps, buildings, bin_ids
    df["timestamp"]   = _make_timestamps(n)
    df["building_id"] = rng.choice(BUILDINGS, size=n)
    df["bin_id"]      = [f"BIN-{b}-{rng.integers(1,6):02d}" for b in df["building_id"]]

    # Inject NaNs into sensor columns
    for col in ["weight_g", "moisture_pct", "metal_signal", "transparency", "acoustic_hardness"]:
        df[col] = _inject_nans(df[col].values)

    # Round for readability
    df["weight_g"]           = df["weight_g"].round(1)
    df["moisture_pct"]       = df["moisture_pct"].round(2)
    df["metal_signal"]       = df["metal_signal"].round(4)
    df["transparency"]       = df["transparency"].round(4)
    df["acoustic_hardness"]  = df["acoustic_hardness"].round(4)

    # Build event_id
    df.sort_values("timestamp", inplace=True, ignore_index=True)
    df.insert(0, "event_id", [f"{id_prefix}-{i+1:05d}" for i in range(len(df))])

    # Column order
    base_cols = ["event_id", "timestamp", "building_id", "bin_id",
                 "weight_g", "moisture_pct", "metal_signal", "transparency", "acoustic_hardness"]
    if include_label:
        return df[base_cols + ["category"]]
    return df[base_cols]


def main() -> None:
    print("Generating waste events …")

    # Train
    train_df = _generate(TRAIN_COUNTS, include_label=True, id_prefix="TRN")
    assert len(train_df) == TRAIN_TOTAL, f"Expected {TRAIN_TOTAL}, got {len(train_df)}"
    class_counts = train_df["category"].value_counts()
    for cls, expected in TRAIN_COUNTS.items():
        actual = class_counts.get(cls, 0)
        assert actual == expected, f"{cls}: expected {expected}, got {actual}"

    train_path = DATA_DIR / "waste_events_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"  ✓ waste_events_train.csv  ({len(train_df)} rows)  → {train_path}")
    print(f"    Class breakdown:\n{class_counts.to_string()}")

    # Test (no labels)
    test_df = _generate(None, include_label=False, id_prefix="TST")
    assert len(test_df) == TEST_TOTAL, f"Expected {TEST_TOTAL}, got {len(test_df)}"
    assert "category" not in test_df.columns, "Test set must not contain labels"

    test_path = DATA_DIR / "waste_events_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  ✓ waste_events_test.csv   ({len(test_df)} rows)  → {test_path}")

    # NaN rate check
    sensor_cols = ["weight_g", "moisture_pct", "metal_signal", "transparency", "acoustic_hardness"]
    for col in sensor_cols:
        rate = train_df[col].isna().mean()
        print(f"    NaN rate [{col}]: {rate:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
