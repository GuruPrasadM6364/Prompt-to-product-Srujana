"""
Phase 4 – Short-horizon Forecasting (6h and 12h ahead).
Targets: electricity_kwh, water_litres, footfall per building.
Baselines: seasonal naive (same hour last week) and last-value.
Model: LightGBM with rolling-origin evaluation (no leakage).
"""
import sys, warnings, json, logging, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DATA_DIR, MODELS_DIR, ALL_BUILDINGS, RANDOM_SEED,
    FORECAST_HORIZONS, FORECAST_TARGETS, SPIKE_PERCENTILE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("forecasting")


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, target: str, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a direct-horizon feature matrix without mixing building histories.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one hour")
    required = {"sampled_hour", target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required forecast columns: {sorted(missing)}")

    df = df.copy()
    df["sampled_hour"] = pd.to_datetime(df["sampled_hour"], errors="raise")
    sort_cols = ["sampled_hour"]
    if "building_id" in df.columns:
        sort_cols.append("building_id")
    df = df.sort_values(sort_cols, kind="stable")
    if "building_id" in df.columns:
        time_deltas = df.groupby("building_id", sort=False)["sampled_hour"].diff().dropna()
        duplicate_keys = df.duplicated(["building_id", "sampled_hour"])
    else:
        time_deltas = df["sampled_hour"].diff().dropna()
        duplicate_keys = df["sampled_hour"].duplicated()
    if duplicate_keys.any():
        raise ValueError("forecast input must contain one row per building and sampled hour")
    if not time_deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("forecast input must have continuous hourly timestamps per building")
    df = df.set_index("sampled_hour")
    buildings = df["building_id"] if "building_id" in df.columns else None

    def shift_by_horizon(values: pd.Series, periods: int) -> pd.Series:
        if buildings is None:
            return values.shift(periods)
        return values.groupby(buildings, sort=False).shift(periods)

    # Calendar features
    df["hour"]        = df.index.hour
    df["dow"]         = df.index.day_of_week
    df["week"]        = df.index.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)

    feature_cols = ["hour", "dow", "week", "is_weekend"]

    # Calendar / event features (merged separately if available)
    for col in ("day_type_encoded", "temperature", "rainfall", "humidity",
                "has_hackathon", "has_exam", "has_alumni_meet", "footfall"):
        if col in df.columns and col != target:
            if col in {"temperature", "rainfall", "humidity", "footfall"}:
                feature_name = f"{col}_lag{horizon}"
                df[feature_name] = shift_by_horizon(df[col], horizon)
                feature_cols.append(feature_name)
            else:
                feature_cols.append(col)

    # Lag features (no leakage: only lags >= horizon)
    max_history = df.groupby(buildings).size().min() if buildings is not None else len(df)
    lags = [h for h in [horizon, horizon+1, horizon+2, horizon+6,
                         24, 48, 24*7]
            if h <= max_history // 2]
    for lag in lags:
        col = f"{target}_lag{lag}"
        df[col] = shift_by_horizon(df[target], lag)
        feature_cols.append(col)

    # Rolling stats (window fully before current row)
    for window in [4, 24, 168]:
        col_mean = f"{target}_roll{window}_mean"
        col_std  = f"{target}_roll{window}_std"
        shifted_target = shift_by_horizon(df[target], horizon)
        if buildings is None:
            df[col_mean] = shifted_target.rolling(window, min_periods=1).mean()
            df[col_std] = shifted_target.rolling(window, min_periods=1).std()
        else:
            grouped_target = shifted_target.groupby(buildings, sort=False)
            df[col_mean] = grouped_target.transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            df[col_std] = grouped_target.transform(
                lambda values: values.rolling(window, min_periods=1).std()
            )
        feature_cols.append(col_mean)
        feature_cols.append(col_std)

    # Building one-hot
    if "building_id" in df.columns:
        dummies = pd.get_dummies(df["building_id"], prefix="bld", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())

    # Drop rows without target
    valid = df[target].notna()
    X = df.loc[valid, [c for c in feature_cols if c in df.columns]].copy()
    if buildings is not None:
        X["building_id"] = buildings.loc[valid]
    y = df.loc[valid, target].copy()

    # Missing lag values at the start of each building's history are unknown,
    # not values to forward-fill from another time or building.
    X = X.fillna(0)

    return X.reset_index(), y.reset_index(drop=True)


# ── Baselines ──────────────────────────────────────────────────────────────────

def seasonal_naive(series: pd.Series, horizon: int, period: int = 168,
                   groups: pd.Series | None = None) -> pd.Series:
    """Same hour last week (period=168 hours)."""
    return series.groupby(groups, sort=False).shift(period) if groups is not None else series.shift(period)


def last_value(series: pd.Series, horizon: int,
               groups: pd.Series | None = None) -> pd.Series:
    """Last observed value (shift by horizon)."""
    return series.groupby(groups, sort=False).shift(horizon) if groups is not None else series.shift(horizon)


def _origin_indices(timestamps: pd.Index, horizon: int, n_origins: int) -> np.ndarray:
    min_train_hours = max(168, len(timestamps) // 4)
    last_origin = len(timestamps) - horizon - 1
    if last_origin < min_train_hours - 1:
        return np.array([], dtype=int)
    return np.unique(np.linspace(
        min_train_hours - 1, last_origin,
        num=min(n_origins, last_origin - min_train_hours + 2),
        dtype=int,
    ))


def _baseline_at_horizon(df: pd.DataFrame, target: str, horizon: int,
                         seasonal: bool) -> pd.Series:
    """Align a historical value to each row by exact building and timestamp."""
    source = df[["sampled_hour", target]].copy()
    if "building_id" in df.columns:
        source["building_id"] = df["building_id"].to_numpy()
        keys = ["building_id", "sampled_hour"]
    else:
        source["building_id"] = "all"
        keys = ["building_id", "sampled_hour"]
    offset = 168 if seasonal else horizon
    source["sampled_hour"] = source["sampled_hour"] + pd.Timedelta(hours=offset)
    source = source.rename(columns={target: "_baseline"})
    aligned = df[["sampled_hour"]].copy()
    aligned["building_id"] = df["building_id"].to_numpy() if "building_id" in df.columns else "all"
    aligned["_row"] = np.arange(len(aligned))
    aligned = aligned.merge(source, on=keys, how="left", sort=False).sort_values("_row")
    return aligned["_baseline"].reset_index(drop=True)


def rolling_baseline_eval(df: pd.DataFrame, target: str, horizon: int,
                          seasonal: bool, n_origins: int = 20) -> Dict[str, float]:
    """Evaluate a baseline on the same direct-horizon origins as the model."""
    sort_cols = ["sampled_hour", "building_id"] if "building_id" in df.columns else ["sampled_hour"]
    df_sorted = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    timestamps = pd.Index(df_sorted["sampled_hour"].drop_duplicates().sort_values())
    origins = _origin_indices(timestamps, horizon, n_origins)
    prediction = _baseline_at_horizon(df_sorted, target, horizon, seasonal)
    maes, smapes = [], []

    for origin_idx in origins:
        forecast_time = timestamps[origin_idx + horizon]
        valid = (df_sorted["sampled_hour"] == forecast_time) & prediction.notna() & df_sorted[target].notna()
        if not valid.any():
            continue
        actual = df_sorted.loc[valid, target].to_numpy()
        predicted = prediction.loc[valid].to_numpy()
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        maes.append(float(np.mean(np.abs(actual - predicted))))
        smapes.append(float(np.mean(np.divide(
            np.abs(actual - predicted), denominator,
            out=np.zeros_like(denominator, dtype=float), where=denominator > 0,
        ))) * 100)

    return {
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "smape": float(np.mean(smapes)) if smapes else float("nan"),
    }


# ── Rolling-origin evaluation ─────────────────────────────────────────────────

def rolling_origin_eval(df: pd.DataFrame, target: str, horizon: int,
                         model_fn, n_origins: int = 20) -> Dict:
    """
    Evaluate direct-horizon forecasts at rolling origins across all buildings.
    """
    sort_cols = ["sampled_hour", "building_id"] if "building_id" in df.columns else ["sampled_hour"]
    df_sorted = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    X_all, y_all = build_features(df_sorted, target, horizon)
    if "building_id" not in X_all.columns:
        X_all["building_id"] = "all"
    timestamps = pd.Index(X_all["sampled_hour"].drop_duplicates().sort_values())
    origin_indices = _origin_indices(timestamps, horizon, n_origins)
    if not len(origin_indices):
        return {
            "mae_mean": float("nan"),
            "smape_mean": float("nan"),
            "spike_precision": 0.0,
            "spike_recall": 0.0,
            "n_origins_used": 0,
            "spike_definition": f">{SPIKE_PERCENTILE}th percentile per building",
        }

    feature_cols = [c for c in X_all.columns if c not in {"sampled_hour", "building_id"}]
    all_errors = {"mae": [], "smape": [], "spike_tp": 0, "spike_fp": 0, "spike_fn": 0}

    for origin_idx in origin_indices:
        origin_time = timestamps[origin_idx]
        forecast_time = timestamps[origin_idx + horizon]
        train_mask = X_all["sampled_hour"] <= origin_time
        valid_mask = X_all["sampled_hour"] == forecast_time
        if not train_mask.any() or not valid_mask.any():
            continue

        model = model_fn()
        model.fit(X_all.loc[train_mask, feature_cols], y_all.loc[train_mask])
        y_true = y_all.loc[valid_mask].to_numpy()
        y_pred = model.predict(X_all.loc[valid_mask, feature_cols])

        mae = float(np.mean(np.abs(y_true - y_pred)))
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        smape = float(np.mean(np.divide(
            np.abs(y_true - y_pred), denom,
            out=np.zeros_like(denom, dtype=float), where=denom > 0,
        )))
        all_errors["mae"].append(mae)
        all_errors["smape"].append(smape * 100)

        valid_buildings = X_all.loc[valid_mask, "building_id"]
        train_buildings = X_all.loc[train_mask, "building_id"]
        train_values = y_all.loc[train_mask]
        thresholds = {
            building: float(train_values.loc[train_buildings == building].quantile(
                SPIKE_PERCENTILE / 100
            ))
            for building in valid_buildings.unique()
        }
        threshold_values = valid_buildings.map(thresholds).to_numpy()
        actual_spike = y_true > threshold_values
        pred_spike = y_pred > threshold_values
        all_errors["spike_tp"] += int(np.sum(actual_spike & pred_spike))
        all_errors["spike_fp"] += int(np.sum(~actual_spike & pred_spike))
        all_errors["spike_fn"] += int(np.sum(actual_spike & ~pred_spike))

    tp = all_errors["spike_tp"]
    fp = all_errors["spike_fp"]
    fn = all_errors["spike_fn"]
    spike_prec = tp / max(tp + fp, 1)
    spike_rec  = tp / max(tp + fn, 1)

    return {
        "mae_mean":    float(np.mean(all_errors["mae"])) if all_errors["mae"] else float("nan"),
        "smape_mean":  float(np.mean(all_errors["smape"])) if all_errors["smape"] else float("nan"),
        "spike_precision": spike_prec,
        "spike_recall":    spike_rec,
        "n_origins_used":  len(all_errors["mae"]),
        "spike_definition": f">{SPIKE_PERCENTILE}th percentile per building",
    }


# ── LightGBM model factory ────────────────────────────────────────────────────

def lgb_factory():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                              num_leaves=31, random_state=RANDOM_SEED,
                              n_jobs=1, verbose=-1)


# ── Full training pipeline ────────────────────────────────────────────────────

def run_forecasting() -> Dict:
    clean_path = DATA_DIR / "hourly_streams_clean.csv"
    if not clean_path.exists():
        log.error(f"Clean hourly data not found at {clean_path}. Run audit first.")
        return {}

    df = pd.read_csv(clean_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["sampled_hour"] = pd.to_datetime(df["sampled_hour"], errors="coerce")

    # Load calendar if available
    cal_path = DATA_DIR / "calendar.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        cal.columns = [c.strip().lower() for c in cal.columns]
        date_col = next((c for c in cal.columns if "date" in c), None)
        if date_col:
            cal["date"] = pd.to_datetime(cal[date_col], errors="coerce")
            df["date"]  = df["sampled_hour"].dt.normalize()
            # Encode day_type
            if "day_type" in cal.columns:
                dt_map = {"working": 0, "weekend": 1, "holiday": 2,
                          "semester_break": 3, "exam": 4}
                cal["day_type_encoded"] = cal["day_type"].map(dt_map).fillna(0).astype(int)
            # Event flags
            if "event" in cal.columns:
                cal["has_hackathon"]  = cal["event"].str.lower().str.contains("hackathon",  na=False).astype(int)
                cal["has_exam"]       = cal["event"].str.lower().str.contains("exam|mid.sem", na=False).astype(int)
                cal["has_alumni_meet"]= cal["event"].str.lower().str.contains("alumni",      na=False).astype(int)
            # Merge
            merge_cols = [c for c in cal.columns if c not in ("date",)]
            df = df.merge(cal[["date"] + [c for c in merge_cols if c in cal.columns]],
                          on="date", how="left")
            df = df.drop(columns=["date"], errors="ignore")

    results = {}

    for horizon in FORECAST_HORIZONS:
        results[f"h{horizon}"] = {}
        for target in FORECAST_TARGETS:
            if target not in df.columns:
                log.warning(f"Target {target} not in data; skipping")
                continue
            log.info(f"\nForecasting {target} at h={horizon}")

            seasonal_result = rolling_baseline_eval(df, target, horizon, seasonal=True)
            last_value_result = rolling_baseline_eval(df, target, horizon, seasonal=False)
            sn_mae = seasonal_result["mae"]
            sn_smape = seasonal_result["smape"]
            lv_mae = last_value_result["mae"]
            lv_smape = last_value_result["smape"]

            log.info(f"  Baseline seasonal-naive MAE={sn_mae:.2f} sMAPE={sn_smape:.1f}%")
            log.info(f"  Baseline last-value    MAE={lv_mae:.2f} sMAPE={lv_smape:.1f}%")

            # LightGBM evaluation
            t_start = time.perf_counter()
            lgb_res = rolling_origin_eval(df, target, horizon, lgb_factory)
            train_time = time.perf_counter() - t_start

            log.info(f"  LightGBM MAE={lgb_res['mae_mean']:.2f} sMAPE={lgb_res['smape_mean']:.1f}% "
                     f"spike_prec={lgb_res['spike_precision']:.3f} spike_rec={lgb_res['spike_recall']:.3f}")

            # Train final model on all data
            t_inf_start = time.perf_counter()
            X_all, y_all = build_features(df, target, horizon)
            X_all = X_all.drop(
                columns=["sampled_hour", "building_id"], errors="ignore"
            ).fillna(0)
            final_model = lgb_factory()
            final_model.fit(X_all, y_all)
            train_done = time.perf_counter() - t_inf_start

            # Single inference latency
            lat_times = []
            x1 = X_all.iloc[:1].copy()
            for _ in range(50):
                t0 = time.perf_counter()
                final_model.predict(x1)
                lat_times.append(time.perf_counter() - t0)
            inf_p50_ms = float(np.percentile(lat_times, 50)) * 1000
            inf_p95_ms = float(np.percentile(lat_times, 95)) * 1000

            # Model size
            import pickle, tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                pickle.dump(final_model, f)
                model_kb = os.path.getsize(f.name) / 1024
                os.unlink(f.name)

            # Save model
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / f"forecast_{target}_h{horizon}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(final_model, f)

            results[f"h{horizon}"][target] = {
                "baseline_seasonal_naive": {"mae": sn_mae, "smape": sn_smape},
                "baseline_last_value":     {"mae": lv_mae, "smape": lv_smape},
                "lightgbm": {
                    "mae":            lgb_res["mae_mean"],
                    "smape":          lgb_res["smape_mean"],
                    "spike_precision":lgb_res["spike_precision"],
                    "spike_recall":   lgb_res["spike_recall"],
                    "spike_def":      lgb_res["spike_definition"],
                    "n_origins":      lgb_res["n_origins_used"],
                    "beats_seasonal_naive": lgb_res["mae_mean"] < sn_mae,
                    "beats_last_value":     lgb_res["mae_mean"] < lv_mae,
                    "train_time_s":  train_time,
                    "inf_p50_ms":    inf_p50_ms,
                    "inf_p95_ms":    inf_p95_ms,
                    "model_size_kb": model_kb,
                },
            }

    # Edge vs central deployment decision
    results["edge_vs_central"] = {
        "decision": "Forecasting runs on central server; edge stores and buffers raw readings.",
        "rationale": (
            "LightGBM inference is fast (<5ms per call) but model training and "
            "feature engineering across 8 buildings require more compute and memory "
            "than a Pi 4 class device provides reliably. The edge node (Pi 4) runs: "
            "(1) waste classification (ONNX, <2s), (2) sensor buffering (SQLite WAL), "
            "(3) last-known-state estimation during outages. The central server runs: "
            "forecasting model training, anomaly detection, dashboard aggregation."
        ),
    }

    # Save results
    out_path = MODELS_DIR / "forecast_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nForecast results saved to {out_path}")
    return results


# ── Offline building forecast (widening uncertainty) ─────────────────────────

def forecast_offline_building(building_id: str, last_known: Dict,
                               hours_offline: int) -> Dict:
    """
    Generate estimates for a building that has been offline.
    Uses last known state + naive seasonal pattern.
    Uncertainty grows with sqrt(hours_offline).
    """
    estimates = {}
    uncertainty_factor = np.sqrt(max(1, hours_offline))
    for target in FORECAST_TARGETS:
        base = last_known.get(target, 0) or 0
        std  = base * 0.15 * uncertainty_factor  # 15% base uncertainty, widening
        estimates[target] = {
            "estimate":       float(base),
            "lower":          float(max(0, base - 2 * std)),
            "upper":          float(base + 2 * std),
            "source":         "last_known_state_extrapolation",
            "hours_offline":  hours_offline,
            "uncertainty_note": f"Uncertainty scaled by sqrt({hours_offline}h offline)",
        }
    return estimates


if __name__ == "__main__":
    run_forecasting()
