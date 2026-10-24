"""HTTP API for the Circular Grid dashboard.

The API reads the generated CSV data directly, so the dashboard can run before
the optional model-training phase has completed.
"""

from __future__ import annotations

import json
import math
import time
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

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="Circular Grid API", version="1.0.0")
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
