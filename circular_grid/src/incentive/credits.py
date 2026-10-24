"""Occupancy-normalized incentive calculations and an auditable credit ledger."""

from __future__ import annotations

from typing import Any

from config import CREDIT_CAP_PER_DAY


class CreditLedger:
    """In-memory ledger for verified and pending building credit awards."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def award(
        self,
        building_id: str,
        date: str,
        occupants: int,
        reason: str,
        raw_credits: float,
        meter_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if occupants <= 0:
            raise ValueError("occupants must be greater than zero")
        if raw_credits < 0:
            raise ValueError("raw_credits cannot be negative")

        already_awarded = self.get_daily_credits(building_id, date)
        daily_cap = occupants * CREDIT_CAP_PER_DAY
        award_amount = min(raw_credits, max(0, daily_cap - already_awarded))
        entry = {
            "building_id": building_id,
            "date": date,
            "occupants": occupants,
            "reason": reason,
            "raw_credits": raw_credits,
            "credits": award_amount,
            "verified": meter_evidence is not None,
            "meter_evidence": meter_evidence,
        }
        self._entries.append(entry)
        return dict(entry)

    def get_daily_credits(self, building_id: str, date: str) -> float:
        return sum(
            entry["credits"]
            for entry in self._entries
            if entry["building_id"] == building_id
            and entry["date"] == date
            and entry["verified"]
        )

    def get_last_entry(self, building_id: str) -> dict[str, Any] | None:
        for entry in reversed(self._entries):
            if entry["building_id"] == building_id:
                return dict(entry)
        return None


def normalize_by_occupancy(savings_kwh: float, occupancy: float) -> float:
    if occupancy <= 0:
        raise ValueError("occupancy must be greater than zero")
    return savings_kwh / occupancy


def detect_load_shift(
    baseline: list[float],
    observed: list[float],
    peak_hours: range = range(8, 19),
    night_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 22, 23),
    peak_drop_threshold: float = 0.4,
    night_increase_threshold: float = 0.5,
) -> dict[str, float | bool]:
    if len(baseline) != 24 or len(observed) != 24:
        raise ValueError("baseline and observed must each contain 24 hourly values")
    if any(value < 0 for value in baseline + observed):
        raise ValueError("hourly usage values cannot be negative")

    def average(values: list[float], hours: range | tuple[int, ...]) -> float:
        return sum(values[hour] for hour in hours) / len(hours)

    baseline_peak = average(baseline, peak_hours)
    observed_peak = average(observed, peak_hours)
    baseline_night = average(baseline, night_hours)
    observed_night = average(observed, night_hours)
    peak_drop = (baseline_peak - observed_peak) / baseline_peak if baseline_peak else 0.0
    night_increase = (
        (observed_night - baseline_night) / baseline_night
        if baseline_night
        else (float("inf") if observed_night else 0.0)
    )
    return {
        "suspicious": peak_drop >= peak_drop_threshold
        and night_increase >= night_increase_threshold,
        "peak_drop_fraction": peak_drop,
        "night_increase_fraction": night_increase,
    }
