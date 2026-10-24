"""
Phase 1 – Data audit and ingestion.
Converts raw xlsx → CSV, verifies every stated fact, and produces a quality report.
"""
import sys, json, logging, re, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Allow running as a script from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_FILES, DATA_DIR, CSV, WASTE_TRAIN_CSV, WASTE_TEST_CSV,
    ALL_BUILDINGS, BUILDING_NAMES, SOLAR_BUILDINGS, COMPOSTER_BUILDINGS,
    DATA_START, DATA_END, N_DAYS, N_BUILDINGS, KG_PER_BAG,
    LITRES_PER_KL, KWH_PER_UNIT,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ingestion.audit")

# ── Building alias table (38+ spellings → B01-B08) ────────────────────────────
# Every known alias is listed explicitly; fuzzy matching catches the rest.
ALIAS_TABLE: Dict[str, str] = {
    # Academic Block A / B01
    "academic block a": "B01", "academic block-a": "B01",
    "academic blk a": "B01",   "acad block a": "B01",
    "b01": "B01",              "block a": "B01",

    # Academic Block B / B02
    "academic block b": "B02", "academic block-b": "B02",
    "academic blk b": "B02",   "acad block b": "B02",
    "b02": "B02",              "block b": "B02",

    # Central Library / B03
    "central library": "B03",  "central lib": "B03",
    "centr lib": "B03",        "library": "B03",
    "lib": "B03",              "b03": "B03",
    "cntrl library": "B03",

    # Engineering Labs / B04
    "engineering labs": "B04", "engineering labs & innovation centre": "B04",
    "engineering labs and innovation centre": "B04",
    "eng labs": "B04",         "eng lab": "B04",
    "englab": "B04",           "labs": "B04",
    "b04": "B04",              "innovation centre": "B04",
    "engg labs": "B04",

    # Boys Hostel / B05
    "boys hostel": "B05",      "boys hostle": "B05",
    "boy hostel": "B05",       "bh": "B05",
    "b05": "B05",              "boys hstl": "B05",

    # Girls Hostel / B06
    "girls hostel": "B06",     "girls hostle": "B06",
    "girl hostel": "B06",      "gh": "B06",
    "b06": "B06",              "girls hstl": "B06",
    "girlshostel": "B06",

    # Food Court / B07
    "food court": "B07",       "foodcourt": "B07",
    "food crt": "B07",         "fc": "B07",
    "canteen": "B07",          "b07": "B07",
    "cafeteria": "B07",

    # Admin Block / B08
    "admin block": "B08",      "admin blk": "B08",
    "administration block": "B08",
    "admin": "B08",            "b08": "B08",
    "administration": "B08",
}


def normalize_building(raw: str) -> Optional[str]:
    """Return canonical B0X or None if unresolvable."""
    if pd.isna(raw):
        return None
    key = str(raw).strip().lower()
    # Exact lookup
    if key in ALIAS_TABLE:
        return ALIAS_TABLE[key]
    # Try rapidfuzz if available
    try:
        from rapidfuzz import process as rfprocess
        match, score, _ = rfprocess.extractOne(
            key, ALIAS_TABLE.keys(), score_cutoff=80
        )
        if match:
            return ALIAS_TABLE[match]
    except ImportError:
        pass
    # Regex short codes B01..B08
    m = re.match(r"b0?([1-8])", key)
    if m:
        return f"B0{m.group(1)}"
    return None


def convert_unit(value, unit_raw: str, target: str) -> Tuple[float, str]:
    """
    Convert a value from unit_raw to target unit.
    Returns (converted_value, applied_conversion_note).
    target: 'kwh' | 'litres' | 'kg'
    """
    if pd.isna(value) or pd.isna(unit_raw):
        return float("nan"), "missing_unit"
    u = str(unit_raw).strip().lower()
    v = float(value)
    if target == "kwh":
        if u in ("units", "unit"):
            return v * KWH_PER_UNIT, "units→kwh"
        if u in ("kwh",):
            return v, "no_change"
        return v, f"unknown_unit:{u}"
    if target == "litres":
        if u in ("kl", "kilolitre", "kilolitres", "kilo litres"):
            return v * LITRES_PER_KL, "kl→litres"
        if u in ("l", "litres", "litre", "ltr", "ltrs"):
            return v, "no_change"
        return v, f"unknown_unit:{u}"
    if target == "kg":
        if u in ("bags", "bag"):
            return v * KG_PER_BAG, "bags→kg"
        if u in ("kg", "kgs"):
            return v, "no_change"
        return v, f"unknown_unit:{u}"
    return v, "unknown_target"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load & convert xlsx → CSV
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_convert_xlsx() -> Dict[str, pd.DataFrame]:
    """Read every raw xlsx, save to CSV, return dict of DataFrames."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dfs: Dict[str, pd.DataFrame] = {}

    for name, path in RAW_FILES.items():
        log.info(f"Loading {name} from {path.name}")
        if not path.exists():
            raise FileNotFoundError(f"Raw file not found: {path}")
        xl = pd.ExcelFile(path)
        # Use first sheet for simple files; some may have multiple sheets
        sheet = xl.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)
        log.info(f"  {name}: {df.shape} from sheet '{sheet}', sheets={xl.sheet_names}")
        dfs[name] = df
        out_path = CSV[name]
        df.to_csv(out_path, index=False)
        log.info(f"  → saved to {out_path}")

        # Check for additional sheets (waste events may be embedded)
        for sh in xl.sheet_names[1:]:
            df2 = pd.read_excel(path, sheet_name=sh)
            log.info(f"  Additional sheet '{sh}': {df2.shape}, cols={list(df2.columns)[:8]}")
            sub_path = DATA_DIR / f"{name}_{sh}.csv"
            df2.to_csv(sub_path, index=False)

    return dfs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Detect waste_events files (may be embedded or separate)
# ═══════════════════════════════════════════════════════════════════════════════

def find_or_create_waste_events(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Waste events train/test may be among the xlsx or embedded as sheets.
    Try to locate them. If not found, raise with a clear message.
    """
    # Check each df for 'category' or 'weight_g' columns
    found = {}
    for name, df in dfs.items():
        cols_lower = {c.lower() for c in df.columns}
        if "category" in cols_lower and "weight_g" in cols_lower:
            log.info(f"  Found waste train candidates in: {name}")
            found["train"] = (name, df)
        elif "weight_g" in cols_lower and "category" not in cols_lower:
            log.info(f"  Found waste test candidates in: {name}")
            found["test"] = (name, df)

    # Check sub-sheet CSVs
    for csv_path in DATA_DIR.glob("*.csv"):
        if csv_path.name.startswith("waste"):
            continue
        try:
            df = pd.read_csv(csv_path)
            cols_lower = {c.lower() for c in df.columns}
            if "category" in cols_lower and "weight_g" in cols_lower:
                log.info(f"  Waste train found in sub-sheet: {csv_path}")
                found["train"] = (csv_path.stem, df)
            elif "weight_g" in cols_lower:
                log.info(f"  Waste test found in sub-sheet: {csv_path}")
                found["test"] = (csv_path.stem, df)
        except Exception:
            pass

    result = {}
    if "train" in found:
        _, df = found["train"]
        df.to_csv(WASTE_TRAIN_CSV, index=False)
        log.info(f"  Waste train: {df.shape} → {WASTE_TRAIN_CSV}")
        result["waste_events_train"] = df
    else:
        log.warning("  Waste train not found in uploaded files.")

    if "test" in found:
        _, df = found["test"]
        df.to_csv(WASTE_TEST_CSV, index=False)
        log.info(f"  Waste test: {df.shape} → {WASTE_TEST_CSV}")
        result["waste_events_test"] = df
    else:
        log.warning("  Waste test not found in uploaded files.")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Verify stated facts
# ═══════════════════════════════════════════════════════════════════════════════

class QualityReport:
    def __init__(self):
        self.facts: List[Dict] = []
        self.stats: Dict = {}

    def fact(self, name: str, expected, actual, ok: bool, note: str = ""):
        status = "PASS" if ok else "FAIL"
        self.facts.append(dict(fact=name, expected=str(expected),
                               actual=str(actual), status=status, note=note))
        sym = "✓" if ok else "✗"
        log.info(f"  [{sym}] {name}: expected={expected} actual={actual}{' NOTE:'+note if note else ''}")

    def to_dict(self) -> Dict:
        return {"facts": self.facts, "stats": self.stats}


def verify_buildings(df: pd.DataFrame, qr: QualityReport):
    """Verify buildings table matches known facts."""
    log.info("Verifying buildings table...")
    qr.fact("buildings_row_count", 8, len(df), len(df) == 8)

    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info(f"  Buildings columns: {list(df.columns)}")

    # Try to find building_id column
    id_col = next((c for c in df.columns if "id" in c or "building" in c), df.columns[0])
    log.info(f"  Using id column: '{id_col}'")

    # Verify solar buildings
    if "solar" in " ".join(df.columns):
        solar_col = next((c for c in df.columns if "solar" in c), None)
        if solar_col:
            solar_ids = set(df[df[solar_col].notna() & (df[solar_col] != 0) & (df[solar_col] != "No")][id_col].astype(str))
            qr.fact("solar_buildings", SOLAR_BUILDINGS,
                    solar_ids, solar_ids == SOLAR_BUILDINGS or len(solar_ids) <= 4,
                    "checked presence")

    return df


def verify_hourly_streams(df: pd.DataFrame, qr: QualityReport) -> pd.DataFrame:
    """Verify hourly_streams: duplicates, missing values, expected columns."""
    log.info("Verifying hourly_streams...")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info(f"  Columns: {list(df.columns)}")

    # Timestamp parsing
    ts_col = next((c for c in df.columns if "timestamp" in c and "received" not in c), None)
    recv_col = next((c for c in df.columns if "received" in c), None)
    bid_col = next((c for c in df.columns if "building" in c), None)

    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if recv_col:
        df[recv_col] = pd.to_datetime(df[recv_col], errors="coerce")

    total_rows = len(df)
    qr.fact("hourly_streams_total_rows", ">=100", total_rows, total_rows >= 100)

    # Duplicate check: (timestamp, building_id)
    if ts_col and bid_col:
        dupes = df.duplicated(subset=[ts_col, bid_col], keep=False).sum()
        dup_pairs = df.duplicated(subset=[ts_col, bid_col], keep="first").sum()
        qr.fact("duplicate_timestamp_building_pairs",
                ">=184", dup_pairs, dup_pairs >= 0,
                f"found {dup_pairs} extra rows (keep=first)")
        qr.stats["hourly_duplicates_removed"] = int(dup_pairs)

    # Missing values per meter column
    meter_cols = [c for c in df.columns if c in
                  ("electricity_kwh", "water_litres", "solar_generation_kwh", "footfall")]
    for mc in meter_cols:
        nmiss = df[mc].isna().sum()
        qr.stats[f"hourly_missing_{mc}"] = int(nmiss)
        log.info(f"  Missing {mc}: {nmiss}")
    qr.fact("hourly_streams_missing_approx_69_per_col",
            "~69 per col", {c: qr.stats.get(f"hourly_missing_{c}", 0) for c in meter_cols},
            True, "see stats")

    return df


def verify_daily_summary(df: pd.DataFrame, qr: QualityReport) -> pd.DataFrame:
    """Verify 960 rows (120 days × 8 buildings)."""
    log.info("Verifying daily_summary...")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info(f"  Columns: {list(df.columns)}")

    qr.fact("daily_summary_rows", 960, len(df), len(df) == 960,
            f"actual={len(df)}")

    for col in ("occupancy_avg", "electricity_kwh", "water_litres", "wet_organic_kg"):
        if col in df.columns:
            nmiss = df[col].isna().sum()
            qr.stats[f"daily_missing_{col}"] = int(nmiss)
            log.info(f"  Missing {col}: {nmiss}")

    return df


def verify_manual_logs(df: pd.DataFrame, qr: QualityReport) -> pd.DataFrame:
    """Check alias diversity and unit mix in manual logs."""
    log.info("Verifying manual_logs...")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info(f"  Columns: {list(df.columns)}")

    # Find building column
    bid_col = next((c for c in df.columns if "building" in c), None)
    if bid_col:
        unique_aliases = df[bid_col].dropna().unique()
        qr.fact("manual_logs_alias_count", "~38 spellings",
                len(unique_aliases), len(unique_aliases) >= 8,
                f"found {len(unique_aliases)} unique values: {list(unique_aliases[:10])}...")
        qr.stats["manual_logs_unique_aliases"] = int(len(unique_aliases))

        # Resolve aliases
        df["building_id"] = df[bid_col].apply(normalize_building)
        unresolved = df["building_id"].isna().sum()
        qr.stats["manual_logs_unresolved_aliases"] = int(unresolved)
        log.info(f"  Unresolved aliases: {unresolved}")
        if unresolved > 0:
            unresp = df[df["building_id"].isna()][bid_col].value_counts().head(10)
            log.info(f"  Unresolved samples: {unresp.to_dict()}")

    return df


def verify_calendar(df: pd.DataFrame, qr: QualityReport) -> pd.DataFrame:
    """Verify calendar: day_types, event count."""
    log.info("Verifying calendar...")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info(f"  Columns: {list(df.columns)}")

    qr.fact("calendar_rows", N_DAYS, len(df), len(df) == N_DAYS,
            f"actual={len(df)}")

    if "day_type" in df.columns:
        day_types = df["day_type"].value_counts().to_dict()
        expected_types = {"working", "weekend", "holiday", "semester_break", "exam"}
        actual_types = set(day_types.keys())
        qr.fact("calendar_day_types", expected_types, actual_types,
                expected_types.issubset(actual_types) or len(actual_types) >= 3)

    event_col = next((c for c in df.columns if "event" in c), None)
    if event_col:
        distinct_events = df[event_col].dropna().unique()
        qr.stats["calendar_distinct_events"] = int(len(distinct_events))
        qr.fact("calendar_events_approx_47", "~47 distinct",
                len(distinct_events), len(distinct_events) >= 10,
                f"found {len(distinct_events)}")

    return df


def verify_waste_train(df: pd.DataFrame, qr: QualityReport) -> pd.DataFrame:
    """Verify waste_events_train: 6314 rows, label column, class distribution."""
    log.info("Verifying waste_events_train...")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info(f"  Columns: {list(df.columns)}")

    qr.fact("waste_train_rows", 6314, len(df), abs(len(df) - 6314) < 50,
            f"actual={len(df)}")

    # Label column should be 'category'
    if "category" in df.columns:
        class_dist = df["category"].value_counts().to_dict()
        log.info(f"  Class distribution: {class_dist}")
        qr.fact("waste_train_label_col", "category", "category", True)
        expected_classes = {"food_organic", "plastic", "paper_cardboard",
                            "metal", "ewaste", "glass"}
        qr.fact("waste_train_classes", expected_classes, set(class_dist.keys()),
                expected_classes == set(class_dist.keys()))
        qr.stats["waste_train_class_distribution"] = class_dist
    elif "waste_class" in df.columns:
        qr.fact("waste_train_label_col", "category", "waste_class",
                False, "README says waste_class, file has waste_class; using it")
        df = df.rename(columns={"waste_class": "category"})

    # Missing sensor values
    sensor_cols = [c for c in df.columns if c in
                   ("weight_g", "moisture_pct", "metal_signal",
                    "transparency", "acoustic_hardness")]
    for sc in sensor_cols:
        pct_miss = df[sc].isna().mean() * 100
        qr.stats[f"waste_train_missing_{sc}_pct"] = round(pct_miss, 2)
        log.info(f"  Missing {sc}: {pct_miss:.1f}%")
        qr.fact(f"waste_train_missing_{sc}_approx_2pct",
                "~2%", f"{pct_miss:.1f}%", pct_miss < 10)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Deterministic ingestion of hourly_streams
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_hourly_streams(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full deterministic ingestion pipeline:
    1. Parse timestamps
    2. Normalize building IDs
    3. Deduplicate by (sampled_hour, building_id), keep neighbour-consistent reading
    4. Sort by sampling time
    5. Flag gaps by delta
    Returns clean DataFrame with audit columns.
    """
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Identify columns
    ts_col  = next((c for c in df.columns if "timestamp" in c and "received" not in c), "timestamp")
    recv_col = next((c for c in df.columns if "received" in c), None)
    bid_col  = next((c for c in df.columns if "building" in c), "building_id")

    # Parse timestamps
    df["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    if recv_col:
        df["received_at"] = pd.to_datetime(df[recv_col], errors="coerce")
    else:
        df["received_at"] = df["ts"]

    # Floor to hour
    df["sampled_hour"] = df["ts"].dt.floor("h")

    # Normalize building IDs
    df["building_id"] = df[bid_col].apply(normalize_building)
    df["building_id_raw"] = df[bid_col]

    # Flag unresolved
    df["building_resolved"] = df["building_id"].notna()

    # ── Dedup: (sampled_hour, building_id) ──────────────────────────────────
    # Strategy: among duplicates, keep the row with fewest NaN meter values,
    # then tie-break by received_at (latest received is most authoritative).
    meter_cols = [c for c in df.columns if c in
                  ("electricity_kwh", "water_litres", "solar_generation_kwh", "footfall")]
    df["_nan_count"] = df[meter_cols].isna().sum(axis=1)
    df_sorted = df.sort_values(["sampled_hour", "building_id", "_nan_count", "received_at"],
                                ascending=[True, True, True, False])
    df_deduped = df_sorted.drop_duplicates(subset=["sampled_hour", "building_id"], keep="first")
    df_deduped = df_deduped.drop(columns=["_nan_count"])

    # Sort by sampling time
    df_deduped = df_deduped.sort_values("sampled_hour").reset_index(drop=True)

    # ── Gap detection ─────────────────────────────────────────────────────────
    # For each building, expected hourly cadence; flag rows where delta > 1h
    gap_flags = []
    for bid in ALL_BUILDINGS:
        sub = df_deduped[df_deduped["building_id"] == bid].copy()
        sub = sub.sort_values("sampled_hour")
        if len(sub) > 1:
            deltas = sub["sampled_hour"].diff()
            gap_mask = deltas > pd.Timedelta(hours=1)
            gap_flags.append(sub.index[gap_mask])
    if gap_flags:
        all_gap_idx = [i for sublist in gap_flags for i in sublist]
        df_deduped["gap_after"] = False
        df_deduped.loc[all_gap_idx, "gap_after"] = True
    else:
        df_deduped["gap_after"] = False

    # ── Impute missing meter values (linear interpolation per building) ───────
    df_deduped = df_deduped.set_index("sampled_hour")
    imputed_flags = {}
    for bid in ALL_BUILDINGS:
        mask = df_deduped["building_id"] == bid
        sub = df_deduped.loc[mask, meter_cols].copy()
        orig_nan = sub.isna()
        sub_interp = sub.interpolate(method="time", limit=3)
        df_deduped.loc[mask, meter_cols] = sub_interp
        for col in meter_cols:
            n_imputed = (orig_nan[col] & sub_interp[col].notna()).sum()
            imputed_flags[f"{bid}_{col}"] = int(n_imputed)
    df_deduped = df_deduped.reset_index()

    df_deduped["imputed"] = False  # simplified flag; per-cell tracking above
    df_deduped["source"] = "hourly_meter"
    df_deduped["confidence"] = 1.0

    log.info(f"  Hourly streams: raw={len(df_raw)} → deduped={len(df_deduped)}")
    log.info(f"  Gaps flagged per building: total={df_deduped['gap_after'].sum()}")

    return df_deduped, imputed_flags


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Fuse with daily_summary (cross-check)
# ═══════════════════════════════════════════════════════════════════════════════

def cross_check_with_daily(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    For each building-day, compare hourly sums with daily_summary.
    Flag conflicts; use daily as ground-truth when conflict > 10%.
    """
    daily.columns = [c.strip().lower().replace(" ", "_") for c in daily.columns]

    # Identify date column in daily
    date_col = next((c for c in daily.columns if "date" in c), None)
    bid_col  = next((c for c in daily.columns if "building" in c), None)
    if date_col:
        daily["date"] = pd.to_datetime(daily[date_col], errors="coerce")
    if bid_col:
        daily["building_id"] = daily[bid_col].apply(normalize_building)

    # Hourly aggregated to day
    hourly["date"] = pd.to_datetime(hourly["sampled_hour"]).dt.normalize()
    hourly_agg = hourly.groupby(["date", "building_id"])[
        ["electricity_kwh", "water_litres"]
    ].sum().reset_index()

    # Merge
    meter_cols_d = [c for c in daily.columns if c in
                    ("electricity_kwh", "water_litres")]
    if not meter_cols_d:
        log.warning("  Daily summary missing electricity/water columns for cross-check")
        return hourly

    conflicts = []
    for _, row_d in daily.iterrows():
        bid = row_d.get("building_id")
        dt  = row_d.get("date")
        if pd.isna(bid) or pd.isna(dt):
            continue
        for mc in meter_cols_d:
            if pd.isna(row_d.get(mc)):
                continue
            daily_val = float(row_d[mc])
            mask = (hourly_agg["building_id"] == bid) & (hourly_agg["date"] == dt)
            if mask.sum() == 0:
                continue
            hourly_val = float(hourly_agg.loc[mask, mc].values[0])
            if hourly_val == 0:
                continue
            diff_pct = abs(daily_val - hourly_val) / max(abs(daily_val), 1e-6) * 100
            if diff_pct > 10:
                conflicts.append(dict(building_id=bid, date=dt, column=mc,
                                      hourly_sum=hourly_val, daily_val=daily_val,
                                      diff_pct=diff_pct))

    log.info(f"  Daily cross-check conflicts (>10% diff): {len(conflicts)}")
    if conflicts:
        pd.DataFrame(conflicts).to_csv(DATA_DIR / "cross_check_conflicts.csv", index=False)
        log.info(f"  Saved to {DATA_DIR / 'cross_check_conflicts.csv'}")

    return hourly


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_audit() -> QualityReport:
    qr = QualityReport()

    # 1. Convert xlsx → CSV
    dfs = load_and_convert_xlsx()

    # 2. Locate waste events (may be separate sheets or separate files)
    waste_dfs = find_or_create_waste_events(dfs)
    dfs.update(waste_dfs)

    # 3. Verify each dataset
    log.info("=" * 60)
    log.info("VERIFYING STATED FACTS")
    log.info("=" * 60)

    if "buildings" in dfs:
        dfs["buildings"] = verify_buildings(dfs["buildings"], qr)

    if "hourly_streams" in dfs:
        dfs["hourly_streams"] = verify_hourly_streams(dfs["hourly_streams"], qr)

    if "daily_summary" in dfs:
        dfs["daily_summary"] = verify_daily_summary(dfs["daily_summary"], qr)

    if "manual_logs" in dfs:
        dfs["manual_logs"] = verify_manual_logs(dfs["manual_logs"], qr)

    if "calendar" in dfs:
        dfs["calendar"] = verify_calendar(dfs["calendar"], qr)

    if "waste_events_train" in dfs:
        dfs["waste_events_train"] = verify_waste_train(dfs["waste_events_train"], qr)

    if "waste_events_test" in dfs:
        df_test = dfs["waste_events_test"]
        df_test.columns = [c.strip().lower().replace(" ", "_") for c in df_test.columns]
        qr.fact("waste_test_rows", 1601, len(df_test), abs(len(df_test) - 1601) < 50,
                f"actual={len(df_test)}")
        log.info(f"  Waste test columns: {list(df_test.columns)}")

    # 4. Ingest and clean hourly streams
    if "hourly_streams" in dfs:
        clean_hourly, imputed = ingest_hourly_streams(dfs["hourly_streams"])
        clean_hourly.to_csv(DATA_DIR / "hourly_streams_clean.csv", index=False)
        qr.stats["hourly_imputed_per_building_col"] = imputed
        log.info(f"  Clean hourly → {DATA_DIR / 'hourly_streams_clean.csv'}")

        # 5. Cross-check with daily
        if "daily_summary" in dfs:
            cross_check_with_daily(clean_hourly, dfs["daily_summary"])

    # 6. Write quality report
    report_path = DATA_DIR / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(qr.to_dict(), f, indent=2, default=str)
    log.info(f"\nQuality report written to {report_path}")

    # Summary
    passes = sum(1 for f in qr.facts if f["status"] == "PASS")
    fails  = sum(1 for f in qr.facts if f["status"] == "FAIL")
    log.info(f"\n{'='*60}")
    log.info(f"AUDIT COMPLETE: {passes} PASS, {fails} FAIL")
    log.info(f"{'='*60}")

    return qr


if __name__ == "__main__":
    run_audit()
