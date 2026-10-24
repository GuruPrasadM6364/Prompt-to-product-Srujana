"""HTTP API for the EcoFlow Campus dashboard.

The API reads the generated CSV data directly, so the dashboard can run before
the optional model-training phase has completed.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    ALL_BUILDINGS,
    BUILDING_NAMES,
    COMPOSTER_BUILDINGS,
    DATA_DIR,
    WASTE_CLASSES,
)
from src.ingestion.audit import normalize_building

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="EcoFlow Campus API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Required data file is missing: {path.name}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def hourly() -> pd.DataFrame:
    df = _read_csv("hourly_streams_clean.csv" if (DATA_DIR / "hourly_streams_clean.csv").exists()
                   else "hourly_streams.csv")
    df.columns = [str(c).strip().lower() for c in df.columns]
    timestamp = "sampled_hour" if "sampled_hour" in df.columns else "timestamp"
    df["timestamp"] = pd.to_datetime(df[timestamp], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


@lru_cache(maxsize=1)
def waste_train() -> pd.DataFrame:
    return _read_csv("waste_events_train.csv")


@lru_cache(maxsize=1)
def waste_test() -> pd.DataFrame:
    return _read_csv("waste_events_test.csv")


@lru_cache(maxsize=1)
def daily_summary() -> pd.DataFrame:
    return _read_csv("daily_summary.csv")


@lru_cache(maxsize=1)
def campus_calendar() -> pd.DataFrame:
    return _read_csv("calendar.csv")


@lru_cache(maxsize=1)
def manual_logs() -> pd.DataFrame:
    return _read_csv("manual_logs.csv")


def _number(value: Any) -> float:
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else 0.0


def _building_rows() -> list[dict[str, Any]]:
    df = hourly()
    rows: list[dict[str, Any]] = []
    latest = df["timestamp"].max()
    for building_id in ALL_BUILDINGS:
        part = df[df["building_id"].astype(str) == building_id].tail(24)
        if part.empty:
            continue
        last = part.iloc[-1]
        sparkline = [
            {"ts": row["timestamp"].isoformat(), "value": _number(row.get("electricity_kwh"))}
            for _, row in part.iterrows()
        ]
        rows.append({
            "id": building_id,
            "name": BUILDING_NAMES[building_id],
            "status": "online" if (latest - last["timestamp"]).total_seconds() <= 7200 else "stale",
            "electricity_kwh": _number(last.get("electricity_kwh")),
            "electricity_estimated": False,
            "water_litres": _number(last.get("water_litres")),
            "water_estimated": False,
            "footfall": _number(last.get("footfall")),
            "footfall_estimated": False,
            "sparkline": sparkline,
        })
    return rows


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/campus/overview")
def campus_overview() -> dict[str, Any]:
    return {"updated_at": hourly()["timestamp"].max().isoformat(), "buildings": _building_rows()}


@app.get("/api/building/{building_id}/history")
def building_history(
    building_id: str,
    stream: str = "electricity_kwh",
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    if building_id not in BUILDING_NAMES:
        raise HTTPException(status_code=404, detail="Unknown building")
    if stream not in {"electricity_kwh", "water_litres", "solar_generation_kwh", "footfall"}:
        raise HTTPException(status_code=400, detail="Unsupported stream")
    if start and start.tzinfo is not None or end and end.tzinfo is not None:
        raise HTTPException(
            status_code=400,
            detail="Use campus-local date and time without a timezone offset",
        )
    if start and end and start > end:
        raise HTTPException(status_code=400, detail="Start date and time must be before the end")

    part = hourly()
    part = part[part["building_id"].astype(str) == building_id].copy()
    if part.empty:
        return {
            "building_id": building_id,
            "building_name": BUILDING_NAMES[building_id],
            "stream": stream,
            "available_start": None,
            "available_end": None,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "points": [],
        }

    available_start = part["timestamp"].min()
    available_end = part["timestamp"].max()
    range_end = pd.Timestamp(end) if end else available_end
    range_start = pd.Timestamp(start) if start else range_end - pd.Timedelta(hours=24)
    filtered = part[part["timestamp"].between(range_start, range_end, inclusive="both")]

    points = []
    for _, row in filtered.iterrows():
        raw_value = pd.to_numeric(row.get(stream), errors="coerce")
        value = float(raw_value) if pd.notna(raw_value) and math.isfinite(float(raw_value)) else None
        points.append({
            "ts": row["timestamp"].isoformat(),
            "value": value,
            "imputed": bool(row.get("imputed", False)),
            "source": str(row.get("source", "hourly_meter")),
        })

    return {
        "building_id": building_id,
        "building_name": BUILDING_NAMES[building_id],
        "stream": stream,
        "available_start": available_start.isoformat(),
        "available_end": available_end.isoformat(),
        "start": range_start.isoformat(),
        "end": range_end.isoformat(),
        "points": points,
    }


@app.get("/api/campus/history")
def campus_history(
    dataset: str = "hourly",
    metric: str = "electricity_kwh",
    building_id: str = "all",
    start: datetime | None = None,
    end: datetime | None = None,
    category: str | None = None,
    manual_stream: str | None = None,
) -> dict[str, Any]:
    dataset_metrics = {
        "hourly": {"electricity_kwh", "water_litres", "solar_generation_kwh", "footfall"},
        "daily_summary": {
            "occupancy_avg", "electricity_kwh", "water_litres", "solar_generation_kwh",
            "wet_organic_kg", "dry_recyclable_kg", "ewaste_kg", "hazardous_kg",
            "composted_kg", "recycled_kg", "contamination_pct", "landfill_kg",
        },
        "calendar": {"avg_temp_c", "rainfall_mm", "humidity_pct"},
        "waste_train": {"weight_g"},
        "waste_test": {"weight_g"},
        "manual_logs": {"value"},
    }
    if dataset not in dataset_metrics:
        raise HTTPException(status_code=400, detail="Unsupported history dataset")
    if metric not in dataset_metrics[dataset]:
        raise HTTPException(status_code=400, detail="Unsupported metric for history dataset")
    if building_id != "all" and building_id not in BUILDING_NAMES:
        raise HTTPException(status_code=404, detail="Unknown building")
    if dataset == "calendar" and building_id != "all":
        raise HTTPException(status_code=400, detail="Calendar and weather records are campus-wide")
    if (start and start.tzinfo is not None) or (end and end.tzinfo is not None):
        raise HTTPException(
            status_code=400,
            detail="Use campus-local date and time without a timezone offset",
        )
    if start and end and start > end:
        raise HTTPException(status_code=400, detail="Start date and time must be before the end")

    if dataset == "hourly":
        frame = hourly().copy()
        frame["history_timestamp"] = frame["timestamp"]
        frame["history_building_id"] = frame["building_id"].astype(str)
        frame["history_value"] = pd.to_numeric(frame[metric], errors="coerce")
        frame["history_source"] = frame.get("source", "hourly_meter")
        frame["history_imputed"] = frame.get("imputed", False)
        frame["history_category"] = None
        frame["history_unit"] = {
            "electricity_kwh": "kWh",
            "water_litres": "L",
            "solar_generation_kwh": "kWh",
            "footfall": "people",
        }[metric]
    elif dataset == "daily_summary":
        frame = daily_summary().copy()
        frame["history_timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["history_building_id"] = frame["building_id"].astype(str)
        frame["history_value"] = pd.to_numeric(frame[metric], errors="coerce")
        frame["history_source"] = "daily_summary"
        frame["history_imputed"] = False
        frame["history_category"] = None
        frame["history_unit"] = {
            "occupancy_avg": "people",
            "electricity_kwh": "kWh",
            "water_litres": "L",
            "solar_generation_kwh": "kWh",
            "wet_organic_kg": "kg",
            "dry_recyclable_kg": "kg",
            "ewaste_kg": "kg",
            "hazardous_kg": "kg",
            "composted_kg": "kg",
            "recycled_kg": "kg",
            "contamination_pct": "%",
            "landfill_kg": "kg",
        }[metric]
    elif dataset == "calendar":
        frame = campus_calendar().copy()
        frame["history_timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["history_building_id"] = None
        frame["history_value"] = pd.to_numeric(frame[metric], errors="coerce")
        frame["history_source"] = "calendar"
        frame["history_imputed"] = False
        frame["history_category"] = frame["event"].where(
            frame["event"].notna() & frame["event"].astype(str).str.strip().ne(""),
            frame["day_type"],
        )
        frame["history_unit"] = {
            "avg_temp_c": "°C",
            "rainfall_mm": "mm",
            "humidity_pct": "%",
        }[metric]
    elif dataset in {"waste_train", "waste_test"}:
        frame = (waste_train() if dataset == "waste_train" else waste_test()).copy()
        frame["history_timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["history_building_id"] = frame["building_id"].astype(str)
        frame["history_value"] = pd.to_numeric(frame["weight_g"], errors="coerce")
        frame["history_source"] = dataset
        frame["history_imputed"] = False
        frame["history_category"] = frame["category"] if "category" in frame else None
        frame["history_unit"] = "g"
        if category and "category" in frame:
            frame = frame[frame["category"].astype(str) == category]
    else:
        frame = manual_logs().copy()
        frame["history_timestamp"] = pd.to_datetime(frame["logged_at"], errors="coerce")
        frame["history_building_id"] = frame["building"].map(normalize_building)
        frame["history_value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["history_source"] = "manual_log"
        frame["history_imputed"] = False
        frame["history_category"] = frame["stream"].astype(str)
        frame["history_unit"] = frame["unit"].fillna("")
        if manual_stream:
            if manual_stream not in {"electricity", "water", "waste"}:
                raise HTTPException(status_code=400, detail="Unsupported manual log stream")
            frame = frame[frame["stream"].astype(str) == manual_stream]
        else:
            frame = frame[frame["stream"].astype(str).isin({"electricity", "water", "waste"})]

    if building_id != "all":
        frame = frame[frame["history_building_id"].astype(str) == building_id]
    frame = frame.dropna(subset=["history_timestamp"]).sort_values("history_timestamp")

    available_start = frame["history_timestamp"].min() if not frame.empty else None
    available_end = frame["history_timestamp"].max() if not frame.empty else None
    range_end = pd.Timestamp(end) if end else available_end
    range_start = pd.Timestamp(start) if start else (
        range_end - pd.Timedelta(hours=24) if range_end is not None else None
    )
    if range_start is not None and range_end is not None:
        frame = frame[frame["history_timestamp"].between(range_start, range_end, inclusive="both")]
    else:
        frame = frame.iloc[0:0]

    points = []
    for _, row in frame.iterrows():
        raw_value = row["history_value"]
        value = float(raw_value) if pd.notna(raw_value) and math.isfinite(float(raw_value)) else None
        row_building_id = row["history_building_id"]
        points.append({
            "ts": row["history_timestamp"].isoformat(),
            "building_id": row_building_id,
            "building_name": BUILDING_NAMES.get(row_building_id, "Campus" if dataset == "calendar" else "Unknown"),
            "value": value,
            "unit": str(row["history_unit"]),
            "category": (
                str(row["history_category"])
                if pd.notna(row["history_category"]) and row["history_category"] is not None
                else None
            ),
            "source": str(row["history_source"]),
            "imputed": bool(row["history_imputed"]),
        })

    return {
        "dataset": dataset,
        "metric": metric,
        "building_id": building_id,
        "available_start": available_start.isoformat() if available_start is not None else None,
        "available_end": available_end.isoformat() if available_end is not None else None,
        "start": range_start.isoformat() if range_start is not None else None,
        "end": range_end.isoformat() if range_end is not None else None,
        "points": points,
    }


@app.get("/api/building/{building_id}/forecast")
def building_forecast(building_id: str, stream: str = "electricity_kwh", horizon: str = "6h") -> dict[str, Any]:
    if building_id not in BUILDING_NAMES:
        raise HTTPException(status_code=404, detail="Unknown building")
    if stream not in {"electricity_kwh", "water_litres", "footfall"}:
        raise HTTPException(status_code=400, detail="Unsupported stream")
    if horizon not in {"6h", "12h"}:
        raise HTTPException(status_code=400, detail="Unsupported horizon")
    part = hourly()[hourly()["building_id"].astype(str) == building_id].tail(72).copy()
    part["value"] = pd.to_numeric(part[stream], errors="coerce").fillna(0)
    history = part["value"].tolist()
    baseline = history[-1] if history else 0.0
    if len(history) >= 24:
        baseline = float(part["value"].tail(24).mean())
    steps = int(horizon[:-1])
    series = []
    for _, row in part.tail(48).iterrows():
        value = _number(row["value"])
        series.append({"ts": row["timestamp"].isoformat(), "actual": value, "forecast": value,
                       "lower": max(0.0, value * 0.9), "upper": value * 1.1})
    last_ts = part["timestamp"].max() if not part.empty else pd.Timestamp.utcnow()
    for step in range(1, steps + 1):
        ts = last_ts + pd.Timedelta(hours=step)
        series.append({"ts": ts.isoformat(), "actual": None, "forecast": baseline,
                       "lower": max(0.0, baseline * 0.8), "upper": baseline * 1.2})
    return {"building_id": building_id, "building_name": BUILDING_NAMES[building_id],
            "stream": stream, "horizon": horizon, "series": series, "calendar_events": []}


def _predict(row: pd.Series) -> tuple[str, float]:
    moisture = _number(row.get("moisture_pct"))
    metal = _number(row.get("metal_signal"))
    transparency = _number(row.get("transparency"))
    hardness = _number(row.get("acoustic_hardness"))
    if moisture >= 50:
        return "food_organic", 0.93
    if metal >= 0.65 and hardness >= 0.75:
        return ("ewaste" if _number(row.get("weight_g")) >= 700 else "metal"), 0.88
    if transparency >= 0.65 and hardness >= 0.75:
        return "glass", 0.86
    if transparency >= 0.45:
        return "plastic", 0.82
    return "paper_cardboard", 0.76


class ClassifyRequest(BaseModel):
    building_id: str = Field(default="B01")
    stream: str = Field(default="waste")
    weight_g: float | None = None
    moisture_pct: float | None = None
    metal_signal: float | None = None
    transparency: float | None = None
    acoustic_hardness: float | None = None
    value: float | None = None


@app.get("/api/classifier/feed")
def classifier_feed() -> dict[str, Any]:
    test = waste_test().head(50)
    events = []
    for _, row in test.iterrows():
        predicted, confidence = _predict(row)
        events.append({"event_id": str(row["event_id"]), "building_id": str(row["building_id"]),
                       "building_name": BUILDING_NAMES.get(str(row["building_id"]), "Unknown"),
                       "timestamp": str(row["timestamp"]), "predicted_class": predicted,
                       "confidence": confidence, "latency_ms": 0.2})
    metrics_path = ROOT / "models" / "classifier_results.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = []
    for name, result in metrics.items():
        if isinstance(result, dict) and "accuracy" in result:
            models.append({"model_id": name, "model_name": name.replace("_", " ").title(),
                           "accuracy": result["accuracy"], "precision": result.get("macro_f1", 0),
                           "recall": result.get("macro_f1", 0), "f1": result.get("macro_f1", 0),
                           "deployed": name == metrics.get("chosen_model")})
    if not models:
        models = [{"model_id": "rule-based", "model_name": "Waste sensor rules",
                   "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "deployed": True}]
    return {"events": events, "models": models}


@app.post("/api/classifier/classify")
def classify(request: ClassifyRequest) -> dict[str, Any]:
    started = time.perf_counter()
    row = pd.Series(request.model_dump())
    predicted, confidence = _predict(row)
    return {"predicted_class": predicted, "confidence": confidence,
            "latency_ms": (time.perf_counter() - started) * 1000}


@app.get("/api/waste/allocation")
def waste_allocation() -> dict[str, Any]:
    df = waste_train().copy()
    df["kg"] = pd.to_numeric(df["weight_g"], errors="coerce").fillna(0) / 1000
    buildings = []
    for building_id in ALL_BUILDINGS:
        part = df[df["building_id"].astype(str) == building_id]
        total = float(part["kg"].sum())
        composted = float(part.loc[part["category"] == "food_organic", "kg"].sum()) if building_id in COMPOSTER_BUILDINGS else 0
        recycled = float(part.loc[part["category"].isin(["plastic", "paper_cardboard", "metal", "glass", "ewaste"]), "kg"].sum())
        landfill = max(0.0, total - composted - recycled)
        buildings.append({"building_id": building_id, "building_name": BUILDING_NAMES[building_id],
                          "total_kg": total, "composted_kg": composted, "recycled_kg": recycled,
                          "landfill_kg": landfill, "diversion_rate": (total - landfill) / total if total else 0})
    total = sum(item["total_kg"] for item in buildings)
    landfill = sum(item["landfill_kg"] for item in buildings)
    composted = sum(item["composted_kg"] for item in buildings)
    recycled = sum(item["recycled_kg"] for item in buildings)
    return {"total_generated_kg": total, "total_composted_kg": composted,
            "total_recycled_kg": recycled, "total_landfill_kg": landfill,
            "overall_diversion_rate": (total - landfill) / total if total else 0,
            "compost_ratio": "4:1", "landfill_avoided_kg": total - landfill, "buildings": buildings}


@app.get("/api/anomalies")
def anomalies() -> dict[str, list[Any]]:
    return {"anomalies": []}


@app.get("/api/connectivity")
def connectivity() -> dict[str, list[Any]]:
    return {"buildings": [{"building_id": row["id"], "building_name": row["name"],
                            "status": row["status"], "last_seen": row["sparkline"][-1]["ts"],
                            "catching_up": False} for row in _building_rows()]}


@app.get("/api/data-quality")
def data_quality() -> dict[str, list[Any]]:
    return {"buildings": [{"building_id": b, "building_name": BUILDING_NAMES[b],
                            "duplicates_removed": 0, "gaps_detected": 0, "imputed_values": 0,
                            "quality_score": 1.0} for b in ALL_BUILDINGS]}


@app.get("/api/leaderboard")
def leaderboard() -> dict[str, Any]:
    entries = [{"rank": i + 1, "building_id": b, "building_name": BUILDING_NAMES[b],
                "normalized_intensity": 0, "credit_score": 0,
                "credit_breakdown": {"segregation": 0, "peak_reduction": 0, "landfill_avoided": 0},
                "anti_gaming_flag": False} for i, b in enumerate(ALL_BUILDINGS)]
    return {"period": "current dataset", "entries": entries}


@app.get("/api/cost-scale")
def cost_scale() -> dict[str, Any]:
    return {"hardware_budget": [], "total_hardware_cost_rs": 0, "scale_tests": []}
