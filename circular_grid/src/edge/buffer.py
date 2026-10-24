"""
src/edge/buffer.py
==================
SQLite WAL-mode edge buffer – one database file per building.

Design
------
* Each building has its own ``<data_dir>/edge_buffer_<building_id>.db``.
* Events are appended with a monotonic, ever-increasing sequence number.
* Writes are idempotent: re-inserting the same (building_id, device_id,
  sample_timestamp) pair is silently ignored via the UNIQUE constraint.
* ``get_pending`` returns every row with sequence_num > last_ack_seq, ordered
  ascending, so the cloud can process them in order.
* ``acknowledge`` soft-marks rows as uploaded up to and including
  ``up_to_seq``; they remain in the table for audit but are excluded from
  future ``get_pending`` results.
* All connections open the database in WAL (Write-Ahead Logging) mode so that
  concurrent readers never block the writer.

Public API
----------
write_event(building_id, data)  → int            (new sequence_num)
get_pending(building_id, last_ack_seq)  → list[dict]
acknowledge(building_id, up_to_seq)  → int        (rows acknowledged)
get_buffer_stats(building_id)  → dict
reset_buffer(building_id)  → None                 (test helper only)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Configuration ─────────────────────────────────────────────────────────────
try:
    from config import DATA_DIR  # type: ignore[import]
except ImportError:
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

BUFFER_DIR = DATA_DIR / "edge_buffers"
BUFFER_DIR.mkdir(parents=True, exist_ok=True)

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS edge_events (
    sequence_num     INTEGER PRIMARY KEY AUTOINCREMENT,
    building_id      TEXT    NOT NULL,
    device_id        TEXT    NOT NULL,
    sample_timestamp TEXT    NOT NULL,        -- ISO-8601 UTC
    payload_json     TEXT    NOT NULL,        -- full event as JSON
    received_at      TEXT    NOT NULL,        -- when buffer wrote this row
    acknowledged     INTEGER NOT NULL DEFAULT 0,  -- 0=pending, 1=acked
    UNIQUE (building_id, device_id, sample_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_pending
    ON edge_events (building_id, acknowledged, sequence_num);
"""

# Thread-local connection cache so each thread gets its own sqlite3 connection
_local = threading.local()
_lock  = threading.Lock()


def _db_path(building_id: str) -> Path:
    """Return the SQLite file path for a given building."""
    return BUFFER_DIR / f"edge_buffer_{building_id}.db"


def _get_conn(building_id: str) -> sqlite3.Connection:
    """Return (or create) a thread-local connection to the building's buffer DB."""
    key = f"conn_{building_id}"
    conn = getattr(_local, key, None)
    if conn is None:
        path = _db_path(building_id)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_DDL)
        conn.commit()
        setattr(_local, key, conn)
    return conn


# ── Public API ─────────────────────────────────────────────────────────────────

def write_event(building_id: str, data: dict[str, Any]) -> int | None:
    """
    Append an event to the building's edge buffer.

    Parameters
    ----------
    building_id : str
        e.g. "B03"
    data : dict
        Must contain at least ``device_id`` and ``sample_timestamp`` keys.
        All other keys are stored verbatim in ``payload_json``.

    Returns
    -------
    int
        The assigned ``sequence_num`` for this event.
    None
        If the event was a duplicate (same building/device/timestamp) and was
        silently ignored.

    Raises
    ------
    ValueError
        If required keys are missing from ``data``.
    """
    required = {"device_id", "sample_timestamp"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"write_event: missing required keys: {missing}")

    device_id       = str(data["device_id"])
    sample_timestamp = str(data["sample_timestamp"])
    received_at     = datetime.now(timezone.utc).isoformat()
    payload_json    = json.dumps(data, default=str)

    conn = _get_conn(building_id)
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO edge_events
                    (building_id, device_id, sample_timestamp, payload_json, received_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (building_id, device_id, sample_timestamp, payload_json, received_at),
            )
            if cur.lastrowid and cur.rowcount == 1:
                return cur.lastrowid
            # Duplicate – return existing sequence_num for caller's reference
            row = conn.execute(
                """
                SELECT sequence_num FROM edge_events
                WHERE building_id = ? AND device_id = ? AND sample_timestamp = ?
                """,
                (building_id, device_id, sample_timestamp),
            ).fetchone()
            return None if row is None else None  # Idempotent: caller ignores dups
    except sqlite3.Error as exc:
        raise RuntimeError(f"write_event failed for {building_id}: {exc}") from exc


def get_pending(building_id: str, last_ack_seq: int = 0) -> list[dict]:
    """
    Return all unacknowledged events with sequence_num > last_ack_seq,
    ordered ascending by sequence_num.

    Parameters
    ----------
    building_id : str
    last_ack_seq : int
        The last sequence number the cloud has confirmed receiving.
        Default 0 returns all pending events from the start.

    Returns
    -------
    list[dict]
        Each dict contains: sequence_num, building_id, device_id,
        sample_timestamp, payload_json (parsed dict), received_at.
    """
    conn = _get_conn(building_id)
    rows = conn.execute(
        """
        SELECT sequence_num, building_id, device_id, sample_timestamp,
               payload_json, received_at
        FROM   edge_events
        WHERE  building_id  = ?
          AND  acknowledged = 0
          AND  sequence_num > ?
        ORDER  BY sequence_num ASC
        """,
        (building_id, last_ack_seq),
    ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        try:
            d["payload"] = json.loads(d.pop("payload_json"))
        except json.JSONDecodeError:
            d["payload"] = {}
        result.append(d)
    return result


def acknowledge(building_id: str, up_to_seq: int) -> int:
    """
    Mark all events with sequence_num <= up_to_seq as acknowledged (uploaded).

    This is idempotent: re-acknowledging already-acked rows is a no-op.

    Parameters
    ----------
    building_id : str
    up_to_seq : int
        All rows with sequence_num ≤ this value will be marked acknowledged.

    Returns
    -------
    int
        Number of rows newly acknowledged in this call.
    """
    conn = _get_conn(building_id)
    with conn:
        cur = conn.execute(
            """
            UPDATE edge_events
            SET    acknowledged = 1
            WHERE  building_id  = ?
              AND  sequence_num <= ?
              AND  acknowledged  = 0
            """,
            (building_id, up_to_seq),
        )
    return cur.rowcount


def get_buffer_stats(building_id: str) -> dict:
    """
    Return summary statistics for a building's buffer.

    Returns
    -------
    dict with keys:
        total_events, pending_events, acknowledged_events,
        min_pending_seq, max_pending_seq, oldest_pending_ts, newest_pending_ts
    """
    conn = _get_conn(building_id)
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                          AS total_events,
            SUM(CASE WHEN acknowledged=0 THEN 1 ELSE 0 END)  AS pending_events,
            SUM(CASE WHEN acknowledged=1 THEN 1 ELSE 0 END)  AS acknowledged_events,
            MIN(CASE WHEN acknowledged=0 THEN sequence_num END) AS min_pending_seq,
            MAX(CASE WHEN acknowledged=0 THEN sequence_num END) AS max_pending_seq,
            MIN(CASE WHEN acknowledged=0 THEN sample_timestamp END) AS oldest_pending_ts,
            MAX(CASE WHEN acknowledged=0 THEN sample_timestamp END) AS newest_pending_ts
        FROM edge_events
        WHERE building_id = ?
        """,
        (building_id,),
    ).fetchone()
    return dict(row) if row else {}


def reset_buffer(building_id: str) -> None:
    """
    **Test helper only** – deletes the buffer DB file for a building.
    Closes the thread-local connection first.
    """
    key = f"conn_{building_id}"
    conn = getattr(_local, key, None)
    if conn:
        try:
            conn.close()
        except Exception:
            pass
        setattr(_local, key, None)
    path = _db_path(building_id)
    if path.exists():
        path.unlink()


# ── Self-test (run directly) ───────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(0)
    BID = "B03"
    reset_buffer(BID)

    # Write 10 events
    seqs = []
    for i in range(10):
        ts = f"2026-07-01T{8 + i:02d}:00:00+00:00"
        seq = write_event(BID, {"device_id": f"SENSOR-{BID}-E", "sample_timestamp": ts,
                                "kwh": round(random.uniform(1, 5), 3)})
        seqs.append(seq)
    print("Inserted seqs:", seqs)

    # Duplicate insertion (should be ignored)
    dup_seq = write_event(BID, {"device_id": f"SENSOR-{BID}-E",
                                 "sample_timestamp": "2026-07-01T08:00:00+00:00",
                                 "kwh": 999})
    print("Duplicate result (expect None):", dup_seq)

    pending = get_pending(BID, last_ack_seq=0)
    print(f"Pending count (expect 10): {len(pending)}")

    ack_count = acknowledge(BID, up_to_seq=5)
    print(f"Acknowledged (expect 5): {ack_count}")

    pending2 = get_pending(BID, last_ack_seq=5)
    print(f"Still pending (expect 5): {len(pending2)}")

    stats = get_buffer_stats(BID)
    print("Stats:", stats)

    reset_buffer(BID)
    print("Self-test passed ✓")
