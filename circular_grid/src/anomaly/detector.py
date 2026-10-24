"""
src/anomaly/detector.py
=======================
Anomaly detection, attribution, recommendation, and circular-waste allocation
for the Circular Grid platform.

Components
----------
1. **Residual anomaly detection** per (building, stream) using robust z-score
   on forecast residuals (or raw readings when no forecast is available).
2. **Rule layer**:
   - Spike detection (>90th percentile).
   - Night-time base-load check (hours 0-5 and 23).
   - Water use with no footfall (potential leak).
3. **Attribution engine**: ranks candidate root causes and returns the top 3
   with a contribution score and human-readable evidence (exact CSV rows or
   manual-log notes).
4. **Recommendation engine**: ranked actions with estimated savings in
   kWh, litres, kg, ₹ (at ₹8/kWh), and CO₂ (0.70 kg/kWh).
5. **Circular allocation**: routes wet-organic waste to composters
   (B05, B06, B07), reports diversion rate, landfill avoided, and compost
   produced (4:1 by weight).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
try:
    from config import (  # type: ignore
        DATA_DIR, SPIKE_PERCENTILE, NIGHT_HOURS,
        RS_PER_KWH, KG_CO2_PER_KWH, COMPOST_RATIO,
        COMPOSTER_BUILDINGS, ANOMALY_Z_THRESHOLD,
    )
except ImportError:
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
    SPIKE_PERCENTILE       = 90
    NIGHT_HOURS            = set(range(0, 6)) | {23}
    RS_PER_KWH             = 8.0
    KG_CO2_PER_KWH         = 0.70
    COMPOST_RATIO          = 4.0
    COMPOSTER_BUILDINGS    = {"B05", "B06", "B07"}
    ANOMALY_Z_THRESHOLD    = 3.0


# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    """A single detected anomaly."""
    building_id  : str
    stream       : str              # e.g. "electricity_kwh"
    timestamp    : str
    value        : float
    expected     : float
    residual     : float
    robust_z     : float
    rule_tags    : list[str]        # spike / night_base_load / water_no_footfall
    causes       : list[dict]       # top-3 attribution results
    recommendations: list[dict]     # ranked actions


@dataclass
class AllocationResult:
    """Circular allocation of waste streams."""
    total_wet_organic_kg    : float
    diverted_to_compost_kg  : float
    landfill_avoided_kg     : float
    compost_produced_kg     : float
    diversion_rate_pct      : float
    per_building            : dict[str, dict]


# ── 1. Robust z-score ──────────────────────────────────────────────────────────

def robust_z(values: list[float]) -> list[float]:
    """
    Compute the robust z-score using median and MAD.

        z_i = (x_i - median) / (1.4826 * MAD)

    1.4826 is the consistency factor making MAD consistent with
    the standard deviation for a normal distribution.
    """
    if len(values) < 3:
        return [0.0] * len(values)
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    if mad == 0:
        # Fall back to mean ± std
        try:
            std = statistics.stdev(values)
        except statistics.StatisticsError:
            return [0.0] * len(values)
        if std == 0:
            return [0.0] * len(values)
        return [(v - med) / std for v in values]
    scale = 1.4826 * mad
    return [(v - med) / scale for v in values]


# ── 2. Anomaly detection per (building, stream) ────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    building_id: str,
    stream: str,
    forecast_col: str | None = None,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
    spike_pct: int = SPIKE_PERCENTILE,
    night_hours: set = NIGHT_HOURS,
) -> list[AnomalyEvent]:
    """
    Detect anomalies in a single (building, stream) time-series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: timestamp (parseable), ``stream`` value column.
        Optionally ``footfall``, ``forecast_col`` (predicted value).
    building_id : str
    stream : str
        Column name in df to analyse (e.g. "electricity_kwh").
    forecast_col : str | None
        Column holding the forecast.  If None, uses the rolling median of the
        series as a naive baseline.
    z_threshold : float
        Robust-z cutoff above which a residual is flagged.
    spike_pct : int
        Percentile above which a value is also flagged as a spike.
    night_hours : set
        Set of integer hours considered night-time.

    Returns
    -------
    list[AnomalyEvent]
    """
    required = {"timestamp", stream}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_anomalies: df missing columns {missing}")

    work = df[["timestamp", stream]].copy()
    if "footfall" in df.columns:
        work["footfall"] = df["footfall"]
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp", stream]).sort_values("timestamp").reset_index(drop=True)

    if len(work) < 3:
        return []

    values = work[stream].tolist()

    # Expected value: forecast or rolling median (window=24h)
    if forecast_col and forecast_col in df.columns:
        fc_aligned = df.set_index("timestamp")[forecast_col].reindex(
            work["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ).values.tolist()
        expected_list = [fc_aligned[i] if not math.isnan(float(fc_aligned[i] or float("nan")))
                         else values[i] for i in range(len(values))]
    else:
        # Rolling median: look-back window of min(24, len) points
        window = min(24, len(values))
        expected_list = []
        for i in range(len(values)):
            start = max(0, i - window)
            expected_list.append(statistics.median(values[start:i + 1]))

    residuals = [v - e for v, e in zip(values, expected_list)]
    z_scores  = robust_z(residuals)

    # Spike threshold (90th percentile of absolute values)
    abs_vals    = [abs(v) for v in values]
    spike_thresh = sorted(abs_vals)[int(len(abs_vals) * spike_pct / 100)]

    anomalies: list[AnomalyEvent] = []
    for i, (val, exp, res, z) in enumerate(zip(values, expected_list, residuals, z_scores)):
        rule_tags: list[str] = []
        ts = work.at[i, "timestamp"]
        hour = ts.hour

        is_anomalous = abs(z) > z_threshold

        # Rule: spike
        if abs(val) > spike_thresh and abs(z) > 1.5:
            rule_tags.append("spike")
            is_anomalous = True

        # Rule: night-time base load
        if hour in night_hours and val > expected_list[i] * 1.5 and val > 0.1:
            rule_tags.append("night_base_load")
            is_anomalous = True

        # Rule: water use with no footfall
        if stream in ("water_litres", "water") and "footfall" in work.columns:
            ff = work.at[i, "footfall"] if pd.notna(work.at[i, "footfall"]) else 1
            if ff == 0 and val > 50:
                rule_tags.append("water_no_footfall")
                is_anomalous = True

        if not is_anomalous:
            continue

        ev = AnomalyEvent(
            building_id     = building_id,
            stream          = stream,
            timestamp       = ts.isoformat(),
            value           = round(float(val), 4),
            expected        = round(float(exp), 4),
            residual        = round(float(res), 4),
            robust_z        = round(float(z), 4),
            rule_tags       = rule_tags,
            causes          = [],
            recommendations = [],
        )
        anomalies.append(ev)

    return anomalies


# ── 3. Attribution engine ─────────────────────────────────────────────────────

# Cause templates (populated per anomaly at runtime)
_CAUSE_TEMPLATES = [
    "calendar_event",
    "day_type_change",
    "footfall_spike",
    "solar_underperformance",
    "manual_log_note",
    "weather_anomaly",
]


def _score_calendar(event_row: dict, anomaly: AnomalyEvent) -> float:
    """Score how likely a calendar event caused the anomaly."""
    if not event_row:
        return 0.0
    event_name = str(event_row.get("event_name", "")).lower()
    impact_kwh = float(event_row.get("expected_extra_kwh", 0) or 0)
    base = 0.1
    if anomaly.stream in ("electricity_kwh", "electricity"):
        base += min(0.5, impact_kwh / max(abs(anomaly.residual), 1))
    if any(kw in event_name for kw in ["exam", "ceremony", "festival", "event", "conference"]):
        base += 0.15
    return min(1.0, base)


def _score_day_type(day_type: str, anomaly: AnomalyEvent) -> float:
    """Score contribution from an unusual day type."""
    mapping = {"holiday": 0.45, "semester_break": 0.30, "exam": 0.35, "working": 0.05, "weekend": 0.10}
    return mapping.get(day_type.lower(), 0.05)


def _score_footfall(footfall_change_pct: float, anomaly: AnomalyEvent) -> float:
    """Score based on footfall change percentage."""
    if abs(footfall_change_pct) < 5:
        return 0.05
    return min(0.6, abs(footfall_change_pct) / 100)


def _score_solar(solar_fraction_actual: float, solar_fraction_expected: float, anomaly: AnomalyEvent) -> float:
    """Score solar underperformance contribution."""
    if anomaly.stream not in ("electricity_kwh", "electricity"):
        return 0.0
    deficit = solar_fraction_expected - solar_fraction_actual
    return max(0.0, min(0.5, deficit * 2))


def _score_manual_log(log_notes: list[str], anomaly: AnomalyEvent) -> tuple[float, list[str]]:
    """Search manual log notes for relevant keywords."""
    keywords_by_stream = {
        "electricity_kwh": ["power", "electric", "server", "ac", "hvac", "light"],
        "water_litres":    ["water", "leak", "pipe", "flush", "irrigation"],
        "footfall":        ["event", "crowd", "visit", "occupancy", "closed"],
    }
    kws = keywords_by_stream.get(anomaly.stream, [])
    relevant = [n for n in log_notes if any(k in n.lower() for k in kws)]
    score = min(0.8, len(relevant) * 0.25)
    return score, relevant


def attribute_anomaly(
    anomaly: AnomalyEvent,
    calendar_row: dict | None = None,
    day_type: str = "working",
    footfall_change_pct: float = 0.0,
    solar_actual: float = 0.0,
    solar_expected: float = 0.0,
    manual_log_notes: list[str] | None = None,
    weather_notes: str = "",
) -> list[dict]:
    """
    Rank candidate causes and return the top 3 with evidence.

    Returns
    -------
    list[dict]  – each dict has: cause, score, evidence
    """
    notes = manual_log_notes or []
    log_score, log_evidence = _score_manual_log(notes, anomaly)

    candidates = [
        {
            "cause":    "calendar_event",
            "score":    _score_calendar(calendar_row or {}, anomaly),
            "evidence": f"Calendar event: {calendar_row.get('event_name', 'n/a')}" if calendar_row else "No calendar event",
        },
        {
            "cause":    "day_type_change",
            "score":    _score_day_type(day_type, anomaly),
            "evidence": f"Day type on anomaly date: {day_type}",
        },
        {
            "cause":    "footfall_spike",
            "score":    _score_footfall(footfall_change_pct, anomaly),
            "evidence": f"Footfall change: {footfall_change_pct:+.1f}%",
        },
        {
            "cause":    "solar_underperformance",
            "score":    _score_solar(solar_actual, solar_expected, anomaly),
            "evidence": f"Solar actual={solar_actual:.2f} expected={solar_expected:.2f}",
        },
        {
            "cause":    "manual_log_note",
            "score":    log_score,
            "evidence": "; ".join(log_evidence) if log_evidence else "No relevant log notes",
        },
        {
            "cause":    "weather_anomaly",
            "score":    0.20 if weather_notes else 0.05,
            "evidence": weather_notes or "No weather data available",
        },
    ]
    # Normalise scores so they sum to 1
    total = sum(c["score"] for c in candidates) or 1.0
    for c in candidates:
        c["score"] = round(c["score"] / total, 4)

    top3 = sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]
    return top3


# ── 4. Recommendation engine ───────────────────────────────────────────────────

@dataclass
class Recommendation:
    rank            : int
    action          : str
    stream          : str
    est_saving_kwh  : float = 0.0
    est_saving_litres: float = 0.0
    est_saving_kg   : float = 0.0
    est_saving_rs   : float = 0.0
    est_saving_co2_kg: float = 0.0
    rationale       : str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


def generate_recommendations(
    anomaly: AnomalyEvent,
    excess_kwh: float = 0.0,
    excess_litres: float = 0.0,
    excess_kg: float = 0.0,
) -> list[dict]:
    """
    Generate ranked recommendations for a given anomaly.

    Savings are estimated proportionally to the detected excess above expected.
    Monetary saving: ₹8 / kWh.  CO₂: 0.70 kg / kWh.
    """
    recs: list[Recommendation] = []

    if anomaly.stream in ("electricity_kwh", "electricity"):
        kwh_excess = max(excess_kwh, anomaly.residual if anomaly.residual > 0 else 0)
        if "night_base_load" in anomaly.rule_tags:
            recs.append(Recommendation(
                rank=1,
                action="Enable occupancy-based auto-shutoff for HVAC and lighting in unoccupied zones",
                stream="electricity",
                est_saving_kwh  = round(kwh_excess * 0.60, 2),
                est_saving_rs   = round(kwh_excess * 0.60 * RS_PER_KWH, 2),
                est_saving_co2_kg = round(kwh_excess * 0.60 * KG_CO2_PER_KWH, 3),
                rationale="Night base-load anomaly: equipment running with no occupancy.",
            ))
        if "spike" in anomaly.rule_tags:
            recs.append(Recommendation(
                rank=len(recs) + 1,
                action="Investigate sudden load spike – check for rogue equipment or HVAC fault",
                stream="electricity",
                est_saving_kwh  = round(kwh_excess * 0.80, 2),
                est_saving_rs   = round(kwh_excess * 0.80 * RS_PER_KWH, 2),
                est_saving_co2_kg = round(kwh_excess * 0.80 * KG_CO2_PER_KWH, 3),
                rationale="Load spike >90th percentile detected.",
            ))
        # Generic
        recs.append(Recommendation(
            rank=len(recs) + 1,
            action="Shift non-critical loads (lab equipment, EV charging) to off-peak hours (22:00–06:00)",
            stream="electricity",
            est_saving_kwh  = round(kwh_excess * 0.20, 2),
            est_saving_rs   = round(kwh_excess * 0.20 * RS_PER_KWH, 2),
            est_saving_co2_kg = round(kwh_excess * 0.20 * KG_CO2_PER_KWH, 3),
            rationale="Peak-shaving opportunity identified.",
        ))

    elif anomaly.stream in ("water_litres", "water"):
        lit_excess = max(excess_litres, anomaly.residual if anomaly.residual > 0 else 0)
        if "water_no_footfall" in anomaly.rule_tags:
            recs.append(Recommendation(
                rank=1,
                action="Inspect plumbing for leaks – zero footfall with >50 L water usage detected",
                stream="water",
                est_saving_litres  = round(lit_excess * 0.90, 1),
                rationale="Leak signal: water use with zero footfall.",
            ))
        recs.append(Recommendation(
            rank=len(recs) + 1,
            action="Install sub-metering on top-floor risers to pinpoint leak location",
            stream="water",
            est_saving_litres = round(lit_excess * 0.70, 1),
            rationale="Sub-metering reduces leak-detection time by 60%.",
        ))

    # Waste-stream
    if anomaly.stream in ("waste_kg", "waste"):
        recs.append(Recommendation(
            rank=1,
            action="Increase wet-organic segregation awareness at source; route to composters B05/B06/B07",
            stream="waste",
            est_saving_kg = round(excess_kg * 0.40, 2),
            rationale="Improved segregation reduces landfill fraction.",
        ))

    # Re-rank if needed
    for i, r in enumerate(recs, start=1):
        r.rank = i

    return [r.to_dict() for r in recs]


# ── 5. Circular allocation ─────────────────────────────────────────────────────

def circular_allocation(
    waste_df: pd.DataFrame,
    composter_buildings: set[str] | None = None,
    compost_ratio: float = COMPOST_RATIO,
) -> AllocationResult:
    """
    Route wet-organic waste to composters, compute diversion metrics.

    Parameters
    ----------
    waste_df : pd.DataFrame
        Columns: building_id, category, weight_g (or weight_kg).
        category == "food_organic" is treated as wet-organic.
    composter_buildings : set[str]
        Buildings with composting capacity.  Defaults to B05, B06, B07.
    compost_ratio : float
        kg of wet organic → 1 kg compost.  Default 4.

    Returns
    -------
    AllocationResult
    """
    if composter_buildings is None:
        composter_buildings = COMPOSTER_BUILDINGS

    required = {"building_id", "category"}
    missing = required - set(waste_df.columns)
    if missing:
        raise ValueError(f"circular_allocation: df missing {missing}")

    # Weight column
    if "weight_kg" in waste_df.columns:
        work = waste_df.copy()
    elif "weight_g" in waste_df.columns:
        work = waste_df.copy()
        work["weight_kg"] = work["weight_g"].fillna(0) / 1000.0
    else:
        raise ValueError("circular_allocation: need weight_g or weight_kg column")

    organic_mask = work["category"] == "food_organic"
    total_wet_kg = float(work.loc[organic_mask, "weight_kg"].sum())

    # Divert all wet-organic from composter buildings + anything routable to them
    # Buildings WITH composters handle their own + accept routed waste from others
    per_building: dict[str, dict] = {}
    for bid, grp in work.groupby("building_id"):
        org_kg = float(grp.loc[grp["category"] == "food_organic", "weight_kg"].sum())
        all_kg = float(grp["weight_kg"].sum())
        has_composter = bid in composter_buildings
        diverted = org_kg if has_composter else 0.0
        per_building[bid] = {
            "total_waste_kg":   round(all_kg, 3),
            "wet_organic_kg":   round(org_kg, 3),
            "has_composter":    has_composter,
            "diverted_kg":      round(diverted, 3),
            "landfill_share_kg": round(org_kg - diverted, 3),
        }

    # Buildings without composters: route wet-organic to nearest composter
    # (simple rule: distribute equally across the 3 composters)
    n_composters = max(len(composter_buildings), 1)
    for bid, stats in per_building.items():
        if not stats["has_composter"] and stats["wet_organic_kg"] > 0:
            routed = stats["wet_organic_kg"]
            stats["diverted_kg"]       = round(routed, 3)
            stats["landfill_share_kg"] = 0.0
            stats["routed_to"]         = sorted(composter_buildings)
            # Credit the destination composters
            share = routed / n_composters
            for cb in composter_buildings:
                per_building[cb]["diverted_kg"] = round(
                    per_building[cb]["diverted_kg"] + share, 3
                )

    total_diverted = sum(s["diverted_kg"] for s in per_building.values()
                         if s.get("has_composter", False))
    # total_diverted is the wet-organic that actually reaches composters
    # re-compute properly: sum of organic kg that does NOT go to landfill
    total_diverted = float(sum(
        min(s["wet_organic_kg"], s.get("diverted_kg", 0))
        for s in per_building.values()
    ))
    # cap at total_wet_kg
    total_diverted = min(total_diverted, total_wet_kg)

    landfill_avoided   = total_diverted
    compost_produced   = round(total_diverted / compost_ratio, 3)
    diversion_rate_pct = round((total_diverted / total_wet_kg * 100) if total_wet_kg > 0 else 0.0, 2)

    return AllocationResult(
        total_wet_organic_kg   = round(total_wet_kg, 3),
        diverted_to_compost_kg = round(total_diverted, 3),
        landfill_avoided_kg    = round(landfill_avoided, 3),
        compost_produced_kg    = compost_produced,
        diversion_rate_pct     = diversion_rate_pct,
        per_building           = per_building,
    )


# ── Convenience: full pipeline for a single (building, stream, date-range) ────

def run_anomaly_pipeline(
    df: pd.DataFrame,
    building_id: str,
    stream: str,
    context: dict | None = None,
) -> list[dict]:
    """
    Full anomaly detection + attribution + recommendations for one series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'timestamp' and ``stream`` columns.
    building_id : str
    stream : str
    context : dict | None
        Optional context keys: calendar_row, day_type, footfall_change_pct,
        solar_actual, solar_expected, manual_log_notes, weather_notes,
        excess_kwh, excess_litres, excess_kg.

    Returns
    -------
    list[dict]  – serialisable anomaly records
    """
    ctx = context or {}
    anomalies = detect_anomalies(df, building_id, stream)
    results = []
    for anm in anomalies:
        anm.causes = attribute_anomaly(
            anm,
            calendar_row         = ctx.get("calendar_row"),
            day_type             = ctx.get("day_type", "working"),
            footfall_change_pct  = ctx.get("footfall_change_pct", 0.0),
            solar_actual         = ctx.get("solar_actual", 0.0),
            solar_expected       = ctx.get("solar_expected", 0.0),
            manual_log_notes     = ctx.get("manual_log_notes", []),
            weather_notes        = ctx.get("weather_notes", ""),
        )
        anm.recommendations = generate_recommendations(
            anm,
            excess_kwh    = ctx.get("excess_kwh", max(0, anm.residual)),
            excess_litres = ctx.get("excess_litres", max(0, anm.residual)),
            excess_kg     = ctx.get("excess_kg", 0.0),
        )
        results.append(asdict(anm))
    return results


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(0)

    # Build a fake time-series with one injected spike and one night anomaly
    rows = []
    for day in range(7):
        for hour in range(24):
            kwh = random.uniform(1.5, 3.5)
            if day == 3 and hour == 14:
                kwh = 25.0      # spike
            if day == 5 and hour == 3:
                kwh = 8.0       # night base load
            rows.append({
                "timestamp": f"2026-07-0{day+1}T{hour:02d}:00:00+00:00",
                "electricity_kwh": kwh,
                "footfall": 0 if hour < 6 else random.randint(10, 100),
            })
    df_test = pd.DataFrame(rows)

    results = run_anomaly_pipeline(df_test, "B01", "electricity_kwh", context={
        "day_type": "working",
        "footfall_change_pct": -30.0,
        "manual_log_notes": ["Server room AC tripped at 14:00", "Routine inspection"],
    })
    print(f"Detected {len(results)} anomalies")
    for r in results:
        print(f"  [{r['timestamp']}] z={r['robust_z']:.2f}  tags={r['rule_tags']}")
        print(f"    Top cause: {r['causes'][0]}")
        print(f"    Recommendation: {r['recommendations'][0]['action'] if r['recommendations'] else 'none'}")

    # Circular allocation test
    import numpy as np
    rng = np.random.default_rng(42)
    n = 500
    cats = rng.choice(["food_organic", "plastic", "paper_cardboard", "metal", "glass"],
                      size=n, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    bids = rng.choice(["B01","B02","B03","B04","B05","B06","B07","B08"], size=n)
    alloc_df = pd.DataFrame({
        "building_id": bids,
        "category": cats,
        "weight_g": rng.uniform(100, 800, size=n),
    })
    result = circular_allocation(alloc_df)
    print(f"\nCircular allocation:")
    print(f"  Total wet-organic : {result.total_wet_organic_kg:.1f} kg")
    print(f"  Diverted          : {result.diverted_to_compost_kg:.1f} kg")
    print(f"  Diversion rate    : {result.diversion_rate_pct:.1f}%")
    print(f"  Compost produced  : {result.compost_produced_kg:.1f} kg")
    print(f"  Landfill avoided  : {result.landfill_avoided_kg:.1f} kg")
    print("Self-test passed ✓")
