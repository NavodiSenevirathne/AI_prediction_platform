"""Simulate real-time production traffic against the prediction API.

Examples:
    python scripts/simulate_traffic.py --url http://localhost:8000 --n 300
    python scripts/simulate_traffic.py --url https://<your-api>.onrender.com --n 500 --drift 0.8 --rate 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES  # noqa: E402
from src.data.generate import generate_customers  # noqa: E402


def stream(url: str, n: int, drift: float, rate: float, seed: int | None = None) -> dict:
    customers = generate_customers(n, drift=drift, seed=seed)[FEATURES].to_dict(orient="records")
    session = requests.Session()
    ok, failed, high_risk = 0, 0, 0
    delay = 1.0 / rate if rate > 0 else 0
    started = time.time()
    for i, customer in enumerate(customers, start=1):
        try:
            r = session.post(f"{url.rstrip('/')}/predict", json=customer, timeout=30)
            r.raise_for_status()
            ok += 1
            high_risk += r.json()["risk_level"] == "high"
        except requests.RequestException as exc:
            failed += 1
            print(f"request {i} failed: {exc}", file=sys.stderr)
        if i % 50 == 0:
            print(f"sent {i}/{n} (ok={ok}, failed={failed})")
        if delay:
            time.sleep(delay)
    return {
        "sent": n,
        "ok": ok,
        "failed": failed,
        "high_risk": high_risk,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream simulated customers to the API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200, help="number of requests")
    parser.add_argument("--drift", type=float, default=0.0, help="0 = normal, 1 = heavy drift")
    parser.add_argument("--rate", type=float, default=10, help="requests per second (0 = max)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    print(stream(args.url, args.n, args.drift, args.rate, args.seed))


if __name__ == "__main__":
    main()
