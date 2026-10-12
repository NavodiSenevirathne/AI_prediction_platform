"""Initial (bootstrap) model training.

Usage:
    python -m src.training.train                  # 20k rows, no drift
    python -m src.training.train --data data/customers.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FEATURES, MODEL_DIR, TARGET
from src.data.generate import generate_customers
from src.training.model import DEFAULT_PARAMS, evaluate, save_production_model, train
from src.training.tracking import tracking_run, tracking_uri

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train")


def run_training(
    df: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
    params: dict | None = None,
    data_source: str = "synthetic",
) -> dict:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
    params = {**DEFAULT_PARAMS, **(params or {})}

    with tracking_run("bootstrap-training", tags={"stage": "initial_training"}) as run:
        model = train(train_df, params)
        metrics = evaluate(model, test_df)
        log.info("Test metrics: %s", metrics)

        run.log_params({**params, "train_rows": len(train_df), "data_source": data_source})
        run.log_metrics(metrics, prefix="test_")
        run.log_model(model, input_example=train_df[FEATURES].head(3), register=True)

        metadata = save_production_model(
            model,
            metrics,
            reference=train_df,
            params=params,
            model_dir=model_dir,
            extra={
                "mlflow_run_id": run.run_id,
                "training_rows": len(train_df),
                "data_source": data_source,
                "promotion_reason": "initial training",
            },
        )
    log.info("Saved model v%s to %s (tracking: %s)", metadata["version"], model_dir, tracking_uri())
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the initial churn model")
    parser.add_argument("--data", type=Path, help="CSV with features + churn column")
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
        source = str(args.data)
    else:
        df = generate_customers(args.n_samples, drift=0.0, seed=args.seed)
        source = "synthetic(drift=0.0)"
    run_training(df, model_dir=args.model_dir, data_source=source)


if __name__ == "__main__":
    main()
