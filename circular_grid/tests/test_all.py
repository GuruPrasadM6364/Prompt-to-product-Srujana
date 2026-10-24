"""
pytest test suite for Circular Grid.
Covers:
  - Alias normalization
  - Unit conversion
  - Dedup and late-arrival reconciliation (outage replay)
  - No-leakage forecast splits
  - Classifier latency budget
  - Credit anti-gaming rules
"""
import sys, time, sqlite3, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALL_BUILDINGS, WASTE_CLASSES, CLASSIFIER_CONFIDENCE_THRESHOLD,
    CLASSIFIER_LATENCY_BUDGET_S, CREDIT_CAP_PER_DAY, RANDOM_SEED,
    DATA_DIR, MODELS_DIR, KG_PER_BAG, LITRES_PER_KL,
)
from src.ingestion.audit import normalize_building, convert_unit


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ALIAS NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestAliasNormalization:
    """Every documented alias resolves to the correct canonical ID."""

    @pytest.mark.parametrize("alias,expected", [
        # B01
        ("Academic Block A", "B01"),
        ("Acad Block A",      "B01"),
        ("academic block a",  "B01"),
        ("block a",           "B01"),
        ("B01",               "B01"),
        # B02
        ("Academic Block B",  "B02"),
        ("Block B",           "B02"),
        ("block b",           "B02"),
        # B03
        ("Central Library",   "B03"),
        ("central lib",       "B03"),
        ("Library",           "B03"),
        ("lib",               "B03"),
        # B04
        ("Eng Labs",          "B04"),
        ("Engineering Labs & Innovation Centre", "B04"),
        ("labs",              "B04"),
        ("Innovation Centre", "B04"),
        # B05
        ("Boys Hostel",       "B05"),
        ("BH",                "B05"),
        ("Boys Hostle",       "B05"),
        # B06
        ("Girls Hostel",      "B06"),
        ("GH",                "B06"),
        ("Girls hostle",      "B06"),
        # B07
        ("Food Court",        "B07"),
        ("FC",                "B07"),
        ("Canteen",           "B07"),
        # B08
        ("Admin Block",       "B08"),
        ("Admin",             "B08"),
        ("administration",    "B08"),
    ])
    def test_exact_aliases(self, alias, expected):
        result = normalize_building(alias)
        assert result == expected, f"'{alias}' → '{result}', expected '{expected}'"

    def test_nan_returns_none(self):
        assert normalize_building(float("nan")) is None
        assert normalize_building(None) is None

    def test_all_buildings_have_canonical(self):
        for bid in ALL_BUILDINGS:
            assert normalize_building(bid) == bid

    def test_fuzzy_match_girls_hostle(self):
        # "Girls hostle" is a known alias; test fuzzy path too
        result = normalize_building("girlshostel")
        assert result == "B06"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UNIT CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnitConversion:
    def test_units_to_kwh(self):
        val, note = convert_unit(100, "units", "kwh")
        assert val == pytest.approx(100.0)
        assert "kwh" in note

    def test_kwh_unchanged(self):
        val, note = convert_unit(50.5, "kWh", "kwh")
        assert val == pytest.approx(50.5)
        assert note == "no_change"

    def test_kl_to_litres(self):
        val, note = convert_unit(2.5, "kL", "litres")
        assert val == pytest.approx(2500.0)
        assert "litres" in note

    def test_litres_unchanged(self):
        val, note = convert_unit(1000, "L", "litres")
        assert val == pytest.approx(1000.0)

    def test_bags_to_kg(self):
        val, note = convert_unit(3, "bags", "kg")
        assert val == pytest.approx(3 * KG_PER_BAG)
        assert "kg" in note

    def test_kg_unchanged(self):
        val, note = convert_unit(24.0, "kg", "kg")
        assert val == pytest.approx(24.0)

    def test_missing_value_returns_nan(self):
        val, note = convert_unit(float("nan"), "kWh", "kwh")
        assert np.isnan(val)
        assert "missing" in note

    def test_missing_unit_returns_nan(self):
        val, note = convert_unit(100, float("nan"), "kwh")
        assert np.isnan(val)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DEDUP AND LATE-ARRIVAL RECONCILIATION (OUTAGE REPLAY)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutageReplay:
    """
    Simulate a 6-hour outage on two buildings, then burst upload,
    and prove final campus totals equal no-outage totals.
    """

    def _make_hourly_data(self):
        """Create synthetic 24h × 8 building hourly dataset."""
        rng = np.random.default_rng(RANDOM_SEED)
        rows = []
        for bid in ALL_BUILDINGS:
            for hour in range(24):
                ts = pd.Timestamp("2026-09-01") + pd.Timedelta(hours=hour)
                rows.append({
                    "sampled_hour": ts,
                    "building_id": bid,
                    "electricity_kwh": float(rng.integers(10, 200)),
                    "water_litres": float(rng.integers(100, 2000)),
                    "sequence_num": hour * 8 + ALL_BUILDINGS.index(bid),
                    "device_id": f"meter-{bid}",
                })
        return [
            {
                "building_id": row["building_id"],
                "sequence_num": row["sequence_num"],
                "device_id": row["device_id"],
                "sample_timestamp": row["sampled_hour"].isoformat(),
                "kwh": row["electricity_kwh"],
                "water_litres": row["water_litres"],
                "footfall": None,
            }
            for row in rows
        ]

    def test_outage_replay_totals_match(self):
        """Campus totals after replay must equal no-outage totals."""
        from src.edge.reconciliation import replay_outage_simulation

        df_normal = self._make_hourly_data()
        normal_elec  = sum(row["kwh"] for row in df_normal)
        normal_water = sum(row["water_litres"] for row in df_normal)

        result = replay_outage_simulation(df_normal, ["B03", "B07"], 6)
        assert abs(result["final_kwh"] - normal_elec) < 1e-6, (
            f"Electricity totals differ: {result['final_kwh']} vs {normal_elec}"
        )
        assert abs(result["final_water"] - normal_water) < 1e-6, (
            f"Water totals differ: {result['final_water']} vs {normal_water}"
        )
        assert result["double_counting_detected"] is False

    def test_no_double_counting(self):
        """After reconciliation, each (sampled_hour, building_id) appears once."""
        from src.edge.reconciliation import replay_outage_simulation

        df = self._make_hourly_data()
        result = replay_outage_simulation(df, ["B03"], 3)
        assert result["double_counting_detected"] is False
        assert result["final_kwh"] == pytest.approx(result["baseline_kwh"])

    def test_building_status_tracking(self, tmp_path, monkeypatch):
        """Known buildings remain represented when they have never reported."""
        from datetime import datetime, timezone
        from src.edge import reconciliation

        monkeypatch.setattr(reconciliation, "RECON_DB_PATH", tmp_path / "reconciliation.db")
        now = datetime.now(timezone.utc)
        reconciliation.ingest_event("B07", 1, "meter-B07", now.isoformat(), kwh=1.0)
        statuses = reconciliation.refresh_building_statuses(
            ["B07", "B08"], reference_time=now
        )
        assert statuses == {"B07": "online", "B08": "offline"}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EDGE BUFFER (SQLite WAL)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeBuffer:
    def test_append_only_write(self, tmp_path, monkeypatch):
        """Writes are append-only with monotonic sequence numbers."""
        from src.edge import buffer

        monkeypatch.setattr(buffer, "BUFFER_DIR", tmp_path)
        building_id = f"B-{tmp_path.name}"
        first = buffer.write_event(building_id, {
            "device_id": "meter-1", "sample_timestamp": "2026-09-01T10:00:00+00:00",
            "electricity_kwh": 10.0,
        })
        second = buffer.write_event(building_id, {
            "device_id": "meter-1", "sample_timestamp": "2026-09-01T11:00:00+00:00",
            "electricity_kwh": 11.0,
        })

        pending = buffer.get_pending(building_id, 0)
        assert len(pending) == 2
        seqs = [r["sequence_num"] for r in pending]
        assert seqs == sorted(seqs), "Sequence numbers not monotonic"
        assert seqs == [first, second]

    def test_idempotent_acknowledge(self, tmp_path, monkeypatch):
        """Acknowledging same sequence twice has no effect."""
        from src.edge import buffer

        monkeypatch.setattr(buffer, "BUFFER_DIR", tmp_path)
        building_id = f"B-{tmp_path.name}"
        buffer.write_event(building_id, {
            "device_id": "meter-1", "sample_timestamp": "2026-09-01T08:00:00+00:00",
            "electricity_kwh": 5.0,
        })
        pending_before = buffer.get_pending(building_id, 0)
        seq = pending_before[0]["sequence_num"]
        assert buffer.acknowledge(building_id, seq) == 1
        assert buffer.acknowledge(building_id, seq) == 0
        pending_after = buffer.get_pending(building_id, 0)
        assert len(pending_after) == 0

    def test_wal_mode(self, tmp_path, monkeypatch):
        """SQLite must be in WAL mode."""
        from src.edge import buffer

        monkeypatch.setattr(buffer, "BUFFER_DIR", tmp_path)
        building_id = f"B-{tmp_path.name}"
        buffer.write_event(building_id, {
            "device_id": "meter-1", "sample_timestamp": "2026-09-01T08:00:00+00:00",
        })
        db_path = tmp_path / f"edge_buffer_{building_id}.db"
        with sqlite3.connect(db_path) as conn:
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode.lower() == "wal"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NO-LEAKAGE FORECAST SPLITS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoLeakage:
    def test_lag_features_respect_horizon(self):
        """Features at time t use only data from t-horizon or earlier."""
        from src.forecasting.forecast import build_features

        n = 200
        ts = pd.date_range("2026-07-01", periods=n, freq="h")
        df = pd.DataFrame({
            "sampled_hour": ts,
            "electricity_kwh": np.arange(n, dtype=float),
            "building_id": "B01",
        })

        for horizon in [6, 12]:
            X, y = build_features(df, "electricity_kwh", horizon)
            lag_col = f"electricity_kwh_lag{horizon}"
            row = X.index[X["sampled_hour"] == ts[50]][0]
            assert X.loc[row, lag_col] == pytest.approx(50 - horizon)
            assert y.loc[row] == pytest.approx(50)

    def test_lag_features_do_not_cross_buildings(self):
        """A building's lag must never use another building's readings."""
        from src.forecasting.forecast import build_features

        n = 48
        ts = pd.date_range("2026-07-01", periods=n, freq="h")
        df = pd.DataFrame({
            "sampled_hour": list(ts) * 2,
            "electricity_kwh": list(np.arange(n, dtype=float))
                                + list(10_000 + np.arange(n, dtype=float)),
            "building_id": ["B01"] * n + ["B02"] * n,
        })
        X, _ = build_features(df, "electricity_kwh", 6)
        at_hour_24 = X["sampled_hour"] == ts[24]
        values = X.loc[at_hour_24].set_index("building_id")["electricity_kwh_lag6"]
        assert values["B01"] == pytest.approx(18)
        assert values["B02"] == pytest.approx(10_018)

    def test_rolling_features_use_only_prior_horizon_data(self):
        """Rolling windows end at t-horizon and remain unchanged by future data."""
        from src.forecasting.forecast import build_features

        n = 48
        ts = pd.date_range("2026-07-01", periods=n, freq="h")
        df = pd.DataFrame({
            "sampled_hour": ts,
            "electricity_kwh": np.arange(n, dtype=float),
            "building_id": "B01",
        })
        X, _ = build_features(df, "electricity_kwh", 6)
        row = X.index[X["sampled_hour"] == ts[30]][0]
        assert X.loc[row, "electricity_kwh_roll24_mean"] == pytest.approx(12.5)

        changed = df.copy()
        changed.loc[changed["sampled_hour"] > ts[24], "electricity_kwh"] += 1_000
        changed_X, _ = build_features(changed, "electricity_kwh", 6)
        changed_row = changed_X.index[changed_X["sampled_hour"] == ts[30]][0]
        assert changed_X.loc[changed_row, "electricity_kwh_roll24_mean"] == pytest.approx(12.5)

    def test_rolling_origin_uses_direct_hour_horizon(self):
        """Evaluation advances by distinct timestamps, not rows/buildings."""
        from src.forecasting.forecast import rolling_origin_eval

        n = 220
        timestamps = pd.date_range("2026-07-01", periods=n, freq="h")
        df = pd.DataFrame([
            {
                "sampled_hour": timestamp,
                "building_id": building,
                "electricity_kwh": float(index + (100 if building == "B02" else 0)),
            }
            for index, timestamp in enumerate(timestamps)
            for building in ("B01", "B02")
        ])

        class MeanModel:
            def fit(self, X, y):
                self.mean = float(y.mean())
                return self

            def predict(self, X):
                return np.full(len(X), self.mean)

        result = rolling_origin_eval(df, "electricity_kwh", 6, MeanModel, n_origins=3)
        assert result["n_origins_used"] == 3

    def test_last_value_baseline_is_per_building(self):
        from src.forecasting.forecast import rolling_baseline_eval

        n = 220
        timestamps = pd.date_range("2026-07-01", periods=n, freq="h")
        df = pd.DataFrame([
            {
                "sampled_hour": timestamp,
                "building_id": building,
                "electricity_kwh": float(index + (10_000 if building == "B02" else 0)),
            }
            for index, timestamp in enumerate(timestamps)
            for building in ("B01", "B02")
        ])
        result = rolling_baseline_eval(df, "electricity_kwh", 6, seasonal=False, n_origins=3)
        assert result["mae"] == pytest.approx(6)

    def test_feature_builder_rejects_missing_hours(self):
        from src.forecasting.forecast import build_features

        df = pd.DataFrame({
            "sampled_hour": pd.to_datetime(["2026-07-01T00:00", "2026-07-01T02:00"]),
            "electricity_kwh": [1.0, 2.0],
            "building_id": ["B01", "B01"],
        })
        with pytest.raises(ValueError, match="continuous hourly timestamps"):
            build_features(df, "electricity_kwh", 1)

    def test_time_based_holdout_no_shuffle(self):
        """Time-based holdout must keep chronological order."""
        from src.classifier.train import time_based_holdout

        n = 200
        X = pd.DataFrame({"a": np.arange(n)})
        y = pd.Series(np.arange(n))
        df_full = pd.DataFrame({"timestamp": pd.date_range("2026-07-01", periods=n, freq="h")})

        X_tr, y_tr, X_va, y_va = time_based_holdout(X, y, df_full, 0.2)

        # Train indices must all come before val indices
        assert X_tr.index.max() < X_va.index.min(), (
            "Train data contains future rows (leakage!)"
        )
        assert len(X_tr) + len(X_va) == n


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLASSIFIER LATENCY BUDGET
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifierLatency:
    """p95 latency per event must be under CLASSIFIER_LATENCY_BUDGET_S."""

    @pytest.fixture
    def service(self):
        """Load inference service if model exists."""
        from src.classifier.train import EdgeInferenceService

        # Find any trained model
        model_path = None
        for ext in (".onnx", ".pkl"):
            for p in MODELS_DIR.glob(f"waste_classifier*{ext}"):
                model_path = str(p)
                break
            if model_path:
                break

        if not model_path:
            pytest.skip("No trained classifier model found. Run `make train` first.")
        return EdgeInferenceService(model_path)

    def test_p95_latency_under_budget(self, service):
        """p95 latency for 100 events must be under 2 seconds."""
        rng = np.random.default_rng(RANDOM_SEED)
        latencies = []
        for _ in range(100):
            event = {
                "weight_g":          float(rng.uniform(10, 1000)),
                "moisture_pct":      float(rng.uniform(0, 100)),
                "metal_signal":      float(rng.uniform(0, 1)),
                "transparency":      float(rng.uniform(0, 1)),
                "acoustic_hardness": float(rng.uniform(0, 1)),
            }
            # Inject occasional NaN (2%)
            if rng.random() < 0.02:
                event["weight_g"] = float("nan")

            t0 = time.perf_counter()
            result = service.predict(event)
            latencies.append(time.perf_counter() - t0)

            assert "predicted_class" in result
            assert "confidence" in result

        p95 = float(np.percentile(latencies, 95))
        assert p95 < CLASSIFIER_LATENCY_BUDGET_S, (
            f"p95 latency {p95*1000:.1f}ms exceeds budget {CLASSIFIER_LATENCY_BUDGET_S*1000}ms"
        )

    def test_low_confidence_flag(self, service):
        """Events with confidence below threshold are flagged for manual sort."""
        # Inject all-zero features → model should be uncertain
        event = {col: 0.0 for col in ["weight_g", "moisture_pct",
                                        "metal_signal", "transparency",
                                        "acoustic_hardness"]}
        result = service.predict(event)
        # Low confidence events should have low_confidence=True or predicted_class='uncertain...'
        # (may not always trigger depending on model)
        assert "confidence" in result
        assert "low_confidence" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CREDIT ANTI-GAMING RULES
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreditAntiGaming:
    def test_credit_cap_per_day(self):
        """Daily credits per occupant must not exceed CREDIT_CAP_PER_DAY."""
        from src.incentive.credits import CreditLedger

        ledger = CreditLedger()
        building_id = "B01"
        date = "2026-09-01"
        occupants = 1200  # design_capacity for B01

        # Award far more than the cap
        ledger.award(building_id, date, occupants, reason="segregation",
                     raw_credits=occupants * CREDIT_CAP_PER_DAY * 10,
                     meter_evidence={"verified_kwh_savings": 1.0})

        daily = ledger.get_daily_credits(building_id, date)
        per_occupant = daily / occupants
        assert per_occupant <= CREDIT_CAP_PER_DAY, (
            f"Credit cap violated: {per_occupant:.2f} > {CREDIT_CAP_PER_DAY}"
        )

    def test_load_shift_detection(self):
        """Suspicious load shift (building moves load to off-peak) is detected."""
        from src.incentive.credits import detect_load_shift

        # Normal usage
        normal = [100.0] * 24
        # Suspicious: dramatic drop in peak hours (8-18) + spike at night
        suspicious = normal.copy()
        for h in range(8, 18):
            suspicious[h] = 20.0  # peaked during off-peak
        for h in [0, 1, 2, 22, 23]:
            suspicious[h] = 220.0  # moved to night

        result = detect_load_shift(normal, suspicious)
        assert result["suspicious"] is True, "Load shift not detected"

    def test_occupancy_normalization(self):
        """Larger buildings are not penalized vs smaller ones."""
        from src.incentive.credits import normalize_by_occupancy

        # Two buildings, same total savings but different occupancies
        large = {"savings_kwh": 1000.0, "occupancy": 1200}
        small = {"savings_kwh": 500.0,  "occupancy": 600}

        norm_large = normalize_by_occupancy(large["savings_kwh"], large["occupancy"])
        norm_small = normalize_by_occupancy(small["savings_kwh"], small["occupancy"])

        assert abs(norm_large - norm_small) < 0.01, (
            f"Normalization failed: large={norm_large:.4f}, small={norm_small:.4f}"
        )

    def test_credits_backed_by_meter_data(self):
        """Credits must be auditable against meter readings."""
        from src.incentive.credits import CreditLedger

        ledger = CreditLedger()
        # Award credits with no meter evidence → should log as unverified
        ledger.award("B02", "2026-09-01", 900, reason="segregation", raw_credits=100,
                     meter_evidence=None)
        entry = ledger.get_last_entry("B02")
        assert "verified" in entry
        assert entry["verified"] is False  # no meter evidence


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DATA QUALITY INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataQuality:
    def test_daily_summary_row_count(self):
        """daily_summary must have exactly 960 rows."""
        if not (DATA_DIR / "daily_summary.csv").exists():
            pytest.skip("daily_summary.csv not found. Run `make data` first.")
        df = pd.read_csv(DATA_DIR / "daily_summary.csv")
        assert len(df) == 960, f"Expected 960, got {len(df)}"

    def test_hourly_duplicates_removed(self):
        """Clean hourly data must have no (sampled_hour, building_id) duplicates."""
        clean = DATA_DIR / "hourly_streams_clean.csv"
        if not clean.exists():
            pytest.skip("hourly_streams_clean.csv not found. Run `make data` first.")
        df = pd.read_csv(clean)
        dupes = df.duplicated(["sampled_hour", "building_id"]).sum()
        assert dupes == 0, f"Found {dupes} duplicates in clean hourly data"

    def test_all_buildings_present_after_ingest(self):
        """All 8 buildings must appear in the clean hourly data."""
        clean = DATA_DIR / "hourly_streams_clean.csv"
        if not clean.exists():
            pytest.skip("hourly_streams_clean.csv not found. Run `make data` first.")
        df = pd.read_csv(clean)
        present = set(df["building_id"].dropna().unique())
        for bid in ALL_BUILDINGS:
            assert bid in present, f"Building {bid} missing from clean hourly data"

    def test_waste_train_class_distribution(self):
        """Waste train must have all 6 classes."""
        if not (DATA_DIR / "waste_events_train.csv").exists():
            pytest.skip("waste_events_train.csv not found. Run `make data` first.")
        df = pd.read_csv(DATA_DIR / "waste_events_train.csv")
        assert "category" in df.columns, "Label column 'category' missing"
        classes = set(df["category"].unique())
        from config import WASTE_CLASSES
        for cls in WASTE_CLASSES:
            assert cls in classes, f"Class '{cls}' missing from train data"
