"""Stores every prediction (inputs, output, latency, model version) in SQLite.

This log is the source for the monitoring endpoints: traffic, latency,
prediction distribution and live data-drift detection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_version INTEGER NOT NULL,
    features TEXT NOT NULL,
    probability REAL NOT NULL,
    prediction INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
"""


class PredictionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def log_many(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO predictions VALUES (:id, :created_at, :model_version, :features,"
                " :probability, :prediction, :latency_ms)",
                [{**r, "features": json.dumps(r["features"])} for r in rows],
            )
            self._conn.commit()

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def count(self) -> int:
        return self._query("SELECT COUNT(*) FROM predictions")[0][0]

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT id, created_at, model_version, features, probability, prediction, latency_ms"
            " FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "model_version": r[2],
                "features": json.loads(r[3]),
                "probability": r[4],
                "prediction": r[5],
                "latency_ms": r[6],
            }
            for r in rows
        ]

    def recent_features(self, limit: int) -> pd.DataFrame:
        return pd.DataFrame([r["features"] for r in self.recent(limit)])

    def summary(self, window_minutes: int = 60) -> dict[str, Any]:
        since = (datetime.now(UTC) - timedelta(minutes=window_minutes)).isoformat()
        rows = self._query(
            "SELECT created_at, probability, prediction, latency_ms, model_version"
            " FROM predictions WHERE created_at >= ? ORDER BY created_at",
            (since,),
        )
        total = self.count()
        if not rows:
            return {
                "total_predictions": total,
                "window_minutes": window_minutes,
                "window_predictions": 0,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "positive_rate": None,
                "avg_probability": None,
                "per_minute": [],
                "probability_histogram": [],
                "versions_served": {},
            }
        df = pd.DataFrame(
            rows, columns=["created_at", "probability", "prediction", "latency_ms", "version"]
        )
        df["minute"] = pd.to_datetime(df["created_at"]).dt.floor("min")
        per_minute = (
            df.groupby("minute")
            .agg(
                requests=("probability", "size"),
                avg_latency_ms=("latency_ms", "mean"),
                positive_rate=("prediction", "mean"),
            )
            .reset_index()
        )
        bins = pd.cut(df["probability"], bins=[i / 10 for i in range(11)], include_lowest=True)
        hist = bins.value_counts(sort=False)
        return {
            "total_predictions": total,
            "window_minutes": window_minutes,
            "window_predictions": int(len(df)),
            "avg_latency_ms": round(float(df["latency_ms"].mean()), 3),
            "p95_latency_ms": round(float(df["latency_ms"].quantile(0.95)), 3),
            "positive_rate": round(float(df["prediction"].mean()), 4),
            "avg_probability": round(float(df["probability"].mean()), 4),
            "per_minute": [
                {
                    "minute": m.isoformat(),
                    "requests": int(r),
                    "avg_latency_ms": round(float(lat), 3),
                    "positive_rate": round(float(p), 4),
                }
                for m, r, lat, p in per_minute.itertuples(index=False)
            ],
            "probability_histogram": [
                {"bin": f"{iv.left:.1f}-{iv.right:.1f}", "count": int(c)}
                for iv, c in hist.items()
            ],
            "versions_served": {str(k): int(v) for k, v in df["version"].value_counts().items()},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
