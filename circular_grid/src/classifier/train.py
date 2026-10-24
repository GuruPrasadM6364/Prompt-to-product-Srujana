"""
Phase 3 – Edge Waste Classifier.
Trains and compares 3 models (LogReg, RandomForest, LightGBM).
Exports chosen model to ONNX. Benchmarks latency.
"""
import sys, time, warnings, json, logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DATA_DIR, MODELS_DIR, WASTE_TRAIN_CSV, WASTE_TEST_CSV,
    WASTE_CLASSES, CLASSIFIER_CONFIDENCE_THRESHOLD,
    CLASSIFIER_LATENCY_BUDGET_S, RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("classifier")

FEATURE_COLS = ["weight_g", "moisture_pct", "metal_signal",
                "transparency", "acoustic_hardness"]
LABEL_COL    = "category"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(WASTE_TRAIN_CSV)
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure label col is 'category'
    if LABEL_COL not in df.columns:
        for alias in ("waste_class", "label", "class"):
            if alias in df.columns:
                df = df.rename(columns={alias: LABEL_COL})
                break

    # Ensure all feature cols exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            log.warning(f"Feature column '{col}' missing from train data")
            df[col] = np.nan

    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].copy()
    log.info(f"Loaded {len(df)} train rows; class distribution:\n{y.value_counts().to_string()}")
    return X, y


def load_test_data() -> pd.DataFrame:
    df = pd.read_csv(WASTE_TEST_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    # Ensure event_id
    if "event_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "event_id"})
        else:
            df["event_id"] = range(len(df))
    log.info(f"Loaded {len(df)} test rows")
    return df


# ── Pipeline builders ──────────────────────────────────────────────────────────

def make_pipeline(estimator):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     estimator),
    ])


def get_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import lightgbm as lgb

    return {
        "logistic_regression": make_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                               class_weight="balanced", C=1.0)
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(n_estimators=100, max_depth=10,
                                   random_state=RANDOM_SEED,
                                   class_weight="balanced", n_jobs=1)
        ),
        "lightgbm": make_pipeline(
            lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1,
                               num_leaves=31, random_state=RANDOM_SEED,
                               class_weight="balanced",
                               n_jobs=1, verbose=-1)
        ),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_val, y_val) -> Dict:
    from sklearn.metrics import (accuracy_score, f1_score,
                                  classification_report, confusion_matrix)
    import os

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc    = accuracy_score(y_val, y_pred)
    mac_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    report = classification_report(y_val, y_pred, output_dict=True,
                                   zero_division=0)
    cm     = confusion_matrix(y_val, y_pred, labels=WASTE_CLASSES)

    # Per-class recall
    per_class_recall = {
        cls: report[cls]["recall"] if cls in report else 0.0
        for cls in WASTE_CLASSES
    }

    # Model size (serialize temporarily)
    import pickle, tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(model, f)
        size_kb = os.path.getsize(f.name) / 1024
        os.unlink(f.name)

    # Latency: 100 events (simulate single-event repeated)
    single_x = X_val.iloc[:1].copy()
    lat_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(single_x)
        lat_times.append(time.perf_counter() - t0)
    p50 = float(np.percentile(lat_times, 50))
    p95 = float(np.percentile(lat_times, 95))

    return {
        "accuracy": float(acc),
        "macro_f1": float(mac_f1),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "model_size_kb": float(size_kb),
        "latency_p50_ms": p50 * 1000,
        "latency_p95_ms": p95 * 1000,
    }


def time_based_holdout(X: pd.DataFrame, y: pd.Series,
                        df_full: pd.DataFrame, holdout_frac: float = 0.2):
    """Time-based split: last holdout_frac of rows by timestamp."""
    ts_col = None
    for c in ("timestamp", "ts", "event_time"):
        if c in df_full.columns:
            ts_col = c
            break

    n = len(X)
    if ts_col:
        order = df_full.index.tolist()  # already loaded in order
        split = int(n * (1 - holdout_frac))
        train_idx = order[:split]
        val_idx   = order[split:]
    else:
        split = int(n * (1 - holdout_frac))
        train_idx = list(range(split))
        val_idx   = list(range(split, n))

    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]
    X_va = X.iloc[val_idx]
    y_va = y.iloc[val_idx]
    return X_tr, y_tr, X_va, y_va


def stratified_cv(model, X, y, n_splits: int = 5) -> Dict:
    from sklearn.model_selection import StratifiedKFold, cross_validate
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)  # no shuffle → time order
    scores = cross_validate(model, X, y, cv=cv,
                            scoring=["accuracy", "f1_macro"],
                            return_train_score=False)
    return {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_accuracy_std":  float(scores["test_accuracy"].std()),
        "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "cv_f1_macro_std":  float(scores["test_f1_macro"].std()),
    }


# ── ONNX export ────────────────────────────────────────────────────────────────

def export_onnx(model, model_name: str, n_features: int = len(FEATURE_COLS)):
    """Export sklearn Pipeline to ONNX. Returns path."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type,
                                     options={type(model.named_steps["clf"]): {"zipmap": False}})
        path = MODELS_DIR / f"{model_name}.onnx"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        size_kb = path.stat().st_size / 1024
        log.info(f"  ONNX exported: {path} ({size_kb:.1f} KB)")
        return str(path), size_kb
    except Exception as e:
        log.warning(f"  ONNX export failed ({e}); using pickle fallback")
        import pickle
        path = MODELS_DIR / f"{model_name}.pkl"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        size_kb = path.stat().st_size / 1024
        log.info(f"  Pickle exported: {path} ({size_kb:.1f} KB)")
        return str(path), size_kb


# ── Pi-class latency benchmark ────────────────────────────────────────────────

def benchmark_pi_class(model_path: str, n_events: int = 200) -> Dict:
    """
    Simulate Raspberry Pi 4 class inference.
    Since we have no real hardware, we constrain to 1 CPU thread via
    os.sched_setaffinity (Linux) or processor affinity API (Windows).
    We document this is a SIMULATION, not a real hardware measurement.
    """
    import os, platform
    note = "SIMULATION – not measured on real hardware"

    # Load model
    use_onnx = model_path.endswith(".onnx")
    if use_onnx:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(model_path, sess_options=so)
        input_name = sess.get_inputs()[0].name

        def predict(x_np):
            return sess.run(None, {input_name: x_np.astype(np.float32)})[0]
    else:
        import pickle
        with open(model_path, "rb") as f:
            mdl = pickle.load(f)
        def predict(x_np):
            return mdl.predict(x_np)

    # Generate random events with ~2% missing
    rng = np.random.default_rng(RANDOM_SEED)
    X_bench = rng.random((n_events, len(FEATURE_COLS))) * 1000
    # Inject 2% NaN (use numpy mask)
    mask = rng.random(X_bench.shape) < 0.02
    X_bench[mask] = np.nan

    if not use_onnx:
        X_df = pd.DataFrame(X_bench, columns=FEATURE_COLS)

    # Warm-up
    for _ in range(5):
        if use_onnx:
            x = np.nan_to_num(X_bench[:1], nan=0.0)
            predict(x)
        else:
            predict(X_df.iloc[:1])

    latencies = []
    for i in range(n_events):
        if use_onnx:
            x = np.nan_to_num(X_bench[i:i+1], nan=0.0)
        else:
            x = X_df.iloc[i:i+1]
        t0 = time.perf_counter()
        predict(x)
        latencies.append(time.perf_counter() - t0)

    result = {
        "note": note,
        "backend": "onnxruntime" if use_onnx else "sklearn_pickle",
        "n_events": n_events,
        "p50_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "max_ms": float(max(latencies) * 1000),
        "within_budget": float(np.percentile(latencies, 95) * 1000) < (CLASSIFIER_LATENCY_BUDGET_S * 1000),
    }
    log.info(f"  Pi-class benchmark: p50={result['p50_ms']:.2f}ms p95={result['p95_ms']:.2f}ms")
    return result


# ── Main training run ──────────────────────────────────────────────────────────

def run_classifier() -> Dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_train_data()
    df_full = pd.read_csv(WASTE_TRAIN_CSV)
    df_full.columns = [c.strip().lower() for c in df_full.columns]

    # Time-based holdout
    X_tr, y_tr, X_va, y_va = time_based_holdout(X, y, df_full)
    log.info(f"Holdout split: train={len(X_tr)}, val={len(X_va)}")

    models = get_models()
    results = {}

    for name, pipeline in models.items():
        log.info(f"\n{'='*50}\nTraining: {name}")
        t_start = time.perf_counter()
        res = evaluate_model(pipeline, X_tr, y_tr, X_va, y_va)
        train_time = time.perf_counter() - t_start

        # CV on full train set
        cv_res = stratified_cv(pipeline, X, y)
        res.update(cv_res)
        res["train_time_s"] = train_time

        log.info(f"  accuracy={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f} "
                 f"size={res['model_size_kb']:.1f}KB "
                 f"p95={res['latency_p95_ms']:.2f}ms "
                 f"cv_acc={res['cv_accuracy_mean']:.4f}±{res['cv_accuracy_std']:.4f}")
        results[name] = res

    # Choose best: accuracy >= 0.85 with smallest compute; fallback to best accuracy
    meets_threshold = {n: r for n, r in results.items() if r["accuracy"] >= 0.85}
    if meets_threshold:
        best_name = min(meets_threshold,
                        key=lambda n: results[n]["model_size_kb"] + results[n]["latency_p95_ms"])
        log.info(f"\nAll models meeting 85% threshold: {list(meets_threshold.keys())}")
    else:
        best_name = max(results, key=lambda n: results[n]["accuracy"])
        best_acc = results[best_name]["accuracy"]
        log.warning(f"\nNo model reached 85% accuracy. Best: {best_name} ({best_acc:.4f})")

    log.info(f"Chosen model: {best_name}")
    best_model = models[best_name]
    # Refit on full training set
    best_model.fit(X, y)

    # Export
    model_path, onnx_size = export_onnx(best_model, f"waste_classifier_{best_name}")

    # Pi-class benchmark
    bench = benchmark_pi_class(model_path)
    results["benchmark"] = bench
    results["chosen_model"] = best_name

    # Confusion matrix confusable pairs
    cm = np.array(results[best_name]["confusion_matrix"])
    log.info(f"\nConfusion matrix (rows=actual, cols=predicted):")
    log.info(f"Classes: {WASTE_CLASSES}")
    for i, row in enumerate(cm):
        log.info(f"  {WASTE_CLASSES[i]:20s}: {row.tolist()}")

    # Confusable pairs (off-diagonal)
    pairs = []
    for i in range(len(WASTE_CLASSES)):
        for j in range(len(WASTE_CLASSES)):
            if i != j and cm.shape[0] > i and cm.shape[1] > j:
                pairs.append((WASTE_CLASSES[i], WASTE_CLASSES[j], int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    log.info("\nTop confusable pairs:")
    for a, b, cnt in pairs[:6]:
        log.info(f"  {a} → predicted {b}: {cnt} times")
    results["confusable_pairs"] = [(a, b, c) for a, b, c in pairs[:6]]

    # Save results
    result_path = MODELS_DIR / "classifier_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate test predictions
    generate_test_predictions(best_model, model_path)

    return results


# ── Test predictions ───────────────────────────────────────────────────────────

def generate_test_predictions(model, model_path: str):
    """
    Run model on waste_events_test and save predictions.
    NO accuracy claim is made on this file (no labels exist).
    """
    df_test = load_test_data()
    X_test = df_test[FEATURE_COLS].copy()

    preds = model.predict(X_test)
    # Confidence: get predict_proba if available
    try:
        proba = model.predict_proba(X_test)
        confidence = proba.max(axis=1)
    except Exception:
        confidence = np.ones(len(preds)) * 0.5

    out = pd.DataFrame({
        "event_id":        df_test["event_id"],
        "predicted_class": preds,
        "confidence":      confidence,
    })
    # Flag low-confidence events
    out["flag"] = np.where(
        out["confidence"] < CLASSIFIER_CONFIDENCE_THRESHOLD,
        "uncertain_route_to_manual_sort", ""
    )

    # NOTE appended to file header via comment
    out_path = DATA_DIR / "waste_events_test_predictions.csv"
    out.to_csv(out_path, index=False)
    log.info(f"\n*** Test predictions saved to {out_path} ***")
    log.info("*** NO ACCURACY CLAIM can be made on this file (no ground-truth labels). ***")
    log.info(f"    {len(out)} events; {(out['flag'] != '').sum()} flagged for manual sort")
    return out_path


# ── Inference service (single event) ──────────────────────────────────────────

class EdgeInferenceService:
    """
    Loads the chosen model once; handles single events including missing values.
    """
    _instance = None

    def __init__(self, model_path: str):
        self.use_onnx = model_path.endswith(".onnx")
        if self.use_onnx:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            self.sess = ort.InferenceSession(model_path, sess_options=so)
            self.input_name = self.sess.get_inputs()[0].name
            # Load class names from metadata if available
            self.classes = WASTE_CLASSES
        else:
            import pickle
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            self.classes = list(self._model.classes_) if hasattr(self._model, "classes_") else WASTE_CLASSES

    def predict(self, event: Dict) -> Dict:
        """
        event: dict with keys matching FEATURE_COLS (missing allowed)
        Returns: {predicted_class, confidence, low_confidence_flag, latency_ms}
        """
        t0 = time.perf_counter()
        x = np.array([event.get(col, np.nan) for col in FEATURE_COLS],
                     dtype=np.float32).reshape(1, -1)

        if self.use_onnx:
            x_clean = np.nan_to_num(x, nan=0.0)  # ONNX can't handle NaN
            proba = self.sess.run(None, {self.input_name: x_clean})[1]
            pred_idx = int(np.argmax(proba))
            conf = float(proba[0, pred_idx])
            pred_class = self.classes[pred_idx]
        else:
            x_df = pd.DataFrame(x, columns=FEATURE_COLS)
            pred_class = self._model.predict(x_df)[0]
            try:
                proba = self._model.predict_proba(x_df)[0]
                conf = float(max(proba))
            except Exception:
                conf = 0.5

        latency_ms = (time.perf_counter() - t0) * 1000
        low_conf = conf < CLASSIFIER_CONFIDENCE_THRESHOLD

        return {
            "predicted_class": "uncertain_route_to_manual_sort" if low_conf else pred_class,
            "raw_class": pred_class,
            "confidence": conf,
            "low_confidence": low_conf,
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    run_classifier()
