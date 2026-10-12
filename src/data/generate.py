"""Synthetic customer-churn data source.

A real company would stream customer events from its CRM / billing systems.
Here we simulate that source so the whole platform can run anywhere, and we
can *control drift*: ``drift=0`` reproduces the original customer base, and
``drift=1`` is a heavily shifted population (more month-to-month contracts,
higher prices, more support calls, heavier usage and a changed relationship
between usage and churn, i.e. concept drift).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORIES, FEATURES, TARGET


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int, drift: float = 0.0, seed: int | None = None) -> pd.DataFrame:
    """Generate ``n`` labelled customer records.

    Args:
        n: number of rows.
        drift: 0.0 (reference population) .. 1.0 (strongly drifted population).
        seed: random seed for reproducibility.
    """
    if not 0.0 <= drift <= 1.0:
        raise ValueError("drift must be between 0 and 1")
    rng = np.random.default_rng(seed)
    d = drift

    age = np.clip(rng.normal(42 - 6 * d, 12, n), 18, 85).round().astype(int)
    tenure = np.clip(rng.exponential(30 * (1 - 0.4 * d), n), 0, 72).round().astype(int)

    contract = rng.choice(
        CATEGORIES["contract_type"],
        size=n,
        p=[0.5 + 0.25 * d, 0.3 - 0.1 * d, 0.2 - 0.15 * d],
    )
    internet = rng.choice(
        CATEGORIES["internet_service"],
        size=n,
        p=[0.35 - 0.15 * d, 0.45 + 0.2 * d, 0.2 - 0.05 * d],
    )
    payment = rng.choice(
        CATEGORIES["payment_method"],
        size=n,
        p=[0.3 - 0.05 * d, 0.3 - 0.05 * d, 0.4 + 0.1 * d],
    )

    base_price = np.select([internet == "fiber", internet == "dsl"], [70.0, 50.0], default=20.0)
    monthly = np.clip(base_price + rng.normal(0, 8, n) + 15 * d, 10, 200).round(2)
    total = (monthly * np.maximum(tenure, 1) * rng.uniform(0.95, 1.05, n)).round(2)
    support_calls = rng.poisson(1.2 + 2.0 * d, n)
    usage = rng.gamma(2.0, 1.5 * (1 + d), n).round(2)

    logit = (
        -1.4
        + 1.8 * (contract == "month_to_month")
        - 1.0 * (contract == "two_year")
        + 0.6 * (internet == "fiber")
        + 0.7 * (payment == "electronic_check")
        - 0.06 * tenure
        + 0.5 * support_calls
        + 0.02 * (monthly - 60)
        - 0.015 * (age - 40)
        # concept drift: heavy users start churning much more in the new market
        + d * 1.5 * (usage > 5)
    )
    churn = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_calls": support_calls,
            "avg_daily_usage_gb": usage,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            TARGET: churn,
        }
    )
    return df[FEATURES + [TARGET]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn data")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/customers.csv"))
    args = parser.parse_args()

    df = generate_customers(args.n, args.drift, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} (churn rate {df[TARGET].mean():.1%})")


if __name__ == "__main__":
    main()
