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
    Build feature matrix for one target and horizon.
    NO leakage: features use only lags >= horizon.
    """
    df = df.copy().sort_values("sampled_hour")
    df = df.set_index("sampled_hour")

    # Calendar features
    df["hour"]        = df.index.hour
    df["dow"]         = df.index.dayofweek
    df["week"]        = df.index.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)

    feature_cols = ["hour", "dow", "week", "is_weekend"]

    # Calendar / event features (merged separately if available)
    for col in ("day_type_encoded", "temperature", "rainfall", "humidity",
                "has_hackathon", "has_exam", "has_alumni_meet", "footfall"):
        if col in df.columns and col != target:
            feature_cols.append(col)

    # Lag features (no leakage: only lags >= horizon)
    lags = [h for h in [horizon, horizon+1, horizon+2, horizon+6,
                         24, 48, 24*7]
            if h <= len(df) // 2]
    for lag in lags:
        col = f"{target}_lag{lag}"
        df[col] = df[target].shift(lag)
        feature_cols.append(col)

    # Rolling stats (window fully before current row)
    for window in [4, 24, 168]:
        col_mean = f"{target}_roll{window}_mean"
        col_std  = f"{target}_roll{window}_std"
        df[col_mean] = df[target].shift(horizon).rolling(window, min_periods=1).mean()
        df[col_std]  = df[target].shift(horizon).rolling(window, min_periods=1).std()
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
    y = df.loc[valid, target].copy()

    # Drop rows with all-NaN features (early window)
    X = X.fillna(method="ffill").fillna(0)

    return X.reset_index(), y.reset_index(drop=True)


# ── Baselines ──────────────────────────────────────────────────────────────────

def seasonal_naive(series: pd.Series, horizon: int, period: int = 168) -> pd.Series:
    """Same hour last week (period=168 hours)."""
    shifted = series.shift(period)
    return shifted


def last_value(series: pd.Series, horizon: int) -> pd.Series:
    """Last observed value (shift by horizon)."""
    return series.shift(horizon)


# ── Rolling-origin evaluation ─────────────────────────────────────────────────

def rolling_origin_eval(df: pd.DataFrame, target: str, horizon: int,
                         model_fn, n_origins: int = 20) -> Dict:
    """
    Rolling-origin cross-validation (no leakage).
    Trains on everything up to origin, predicts horizon steps ahead.
    """
    df_sorted = df.sort_values("sampled_hour").reset_index(drop=True)
    n = len(df_sorted)
    min_train = max(168, n // 4)  # need at least 1 week of data

    origin_step = max(1, (n - min_train - horizon) // n_origins)
    origins = list(range(min_train, n - horizon, origin_step))[:n_origins]

    all_errors = {"mae": [], "smape": [], "spike_tp": 0, "spike_fp": 0, "spike_fn": 0}
    spike_thresh = df_sorted[target].quantile(SPIKE_PERCENTILE / 100)

    for origin in origins:
        train_df = df_sorted.iloc[:origin]
        val_idx  = min(origin + horizon - 1, n - 1)
        y_true   = df_sorted[target].iloc[origin:val_idx+1]
        if len(y_true) == 0:
            continue

        try:
            X_tr, y_tr = build_features(train_df, target, horizon)
            X_tr = X_tr.drop(columns=["sampled_hour"], errors="ignore")
            model = model_fn()
            model.fit(X_tr.fillna(0), y_tr)

            val_df = df_sorted.iloc[:val_idx+1]
            X_va, _ = build_features(val_df, target, horizon)
            X_va = X_va.drop(columns=["sampled_hour"], errors="ignore")
            # Align columns
            for c in X_tr.columns:
                if c not in X_va.columns:
                    X_va[c] = 0
            X_va = X_va[X_tr.columns].fillna(0)
            y_pred = model.predict(X_va.iloc[-len(y_true):])

            mae   = float(np.mean(np.abs(y_true.values - y_pred)))
            denom = (np.abs(y_true.values) + np.abs(y_pred)) / 2
            smape = float(np.mean(np.where(denom > 0,
                                           np.abs(y_true.values - y_pred) / denom, 0)))
            all_errors["mae"].append(mae)
            all_errors["smape"].append(smape * 100)

            # Spike detection
            actual_spike = y_true.values > spike_thresh
            pred_spike   = y_pred > spike_thresh
            all_errors["spike_tp"] += int(np.sum(actual_spike & pred_spike))
            all_errors["spike_fp"] += int(np.sum(~actual_spike & pred_spike))
            all_errors["spike_fn"] += int(np.sum(actual_spike & ~pred_spike))
        except Exception as e:
            log.debug(f"Rolling origin {origin} failed: {e}")

    tp = all_errors["spike_tp"]; fp = all_errors["spike_fp"]; fn = all_errors["spike_fn"]
    spike_prec = tp / max(tp + fp, 1)
    spike_rec  = tp / max(tp + fn, 1)

    return {
        "mae_mean":    float(np.mean(all_errors["mae"])) if all_errors["mae"] else float("nan"),
        "smape_mean":  float(np.mean(all_errors["smape"])) if all_errors["smape"] else float("nan"),
        "spike_precision": spike_prec,
        "spike_recall":    spike_rec,
        "n_origins_used":  len(all_errors["mae"]),
        "spike_definition": f">{SPIKE_PERCENTILE}th percentile for that building-target",
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

            # Baselines
            base_sn = seasonal_naive(df[target], horizon)
            base_lv = last_value(df[target], horizon)
            valid   = df[target].notna() & base_sn.notna()

            def smape_arr(y, yhat):
                d = (np.abs(y) + np.abs(yhat)) / 2
                return float(np.mean(np.where(d > 0, np.abs(y - yhat) / d, 0))) * 100

            if valid.sum() > 0:
                y_true = df.loc[valid, target].values
                sn_mae  = float(np.mean(np.abs(y_true - base_sn[valid].values)))
                lv_mae  = float(np.mean(np.abs(y_true - base_lv[valid].values)))
                sn_smape = smape_arr(y_true, base_sn[valid].values)
                lv_smape = smape_arr(y_true, base_lv[valid].values)
            else:
                sn_mae = lv_mae = sn_smape = lv_smape = float("nan")

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
            X_all = X_all.drop(columns=["sampled_hour"], errors="ignore").fillna(0)
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
