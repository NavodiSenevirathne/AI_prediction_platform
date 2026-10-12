"""Automated retraining pipeline (champion / challenger).

Runs on a schedule in GitHub Actions (see .github/workflows/retrain.yml) and can
also be run locally:

    python -m src.pipelines.retrain                 # drift level based on date
    python -m src.pipelines.retrain --drift 0.8     # simulate a strong shift
    python -m src.pipelines.retrain --force         # retrain even without drift

Steps
-----
1. Ingest  - collect the latest labelled production data (CSV if given,
             otherwise the simulated data source).
2. Validate- schema + basic data-quality checks.
3. Monitor - data-drift check (PSI / KS / chi2) against the reference data of
             the current production model, and performance check of the
             production model on the new data.
4. Decide  - retrain if data drift, performance degradation, or --force.
5. Train   - train a challenger on recent data + part of the reference data.
6. Compare - evaluate champion and challenger on the same fresh hold-out set.
7. Promote - only if the challenger is better: write it to models/, which the
             CD workflow then ships to production.

Every run is logged to MLflow and summarised in reports/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORIES,
    FEATURES,
    MAX_AUC_DROP,
    MIN_ACCEPTABLE_AUC,
    MODEL_DIR,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET,
)
from src.data.generate import generate_customers
from src.monitoring.drift import detect_drift
from src.training.model import evaluate, load_production_model, save_production_model, train
from src.training.tracking import tracking_run

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("retrain")

# The simulated market starts drifting from the project start date and is
# fully drifted after DRIFT_HORIZON_DAYS. This makes scheduled runs see
# gradually changing data, like a real business would.
PROJECT_START = date(2026, 10, 2)
DRIFT_HORIZON_DAYS = 21
MIN_IMPROVEMENT = 0.001


def drift_level_for(day: date) -> float:
    return round(min(max((day - PROJECT_START).days / DRIFT_HORIZON_DAYS, 0.0), 1.0), 3)


# --------------------------------------------------------------------------- steps
def ingest(data_path: Path | None, drift: float, n: int, seed: int) -> tuple[pd.DataFrame, str]:
    if data_path:
        return pd.read_csv(data_path), str(data_path)
    return generate_customers(n, drift=drift, seed=seed), f"synthetic(drift={drift})"


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality problems (empty list = data is OK)."""
    problems: list[str] = []
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if len(df) < 500:
        problems.append(f"too few rows: {len(df)}")
    null_share = df[FEATURES + [TARGET]].isna().mean()
    for col, share in null_share.items():
        if share > 0.05:
            problems.append(f"{col}: {share:.1%} missing values")
    for col in NUMERIC_FEATURES:
        if (df[col] < 0).any():
            problems.append(f"{col}: negative values")
    for col, allowed in CATEGORIES.items():
        unknown = set(df[col].dropna().unique()) - set(allowed)
        if unknown:
            problems.append(f"{col}: unknown categories {sorted(unknown)}")
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        problems.append("target must be 0/1")
    return problems


def _write_github_outputs(outputs: dict[str, Any], summary_md: str) -> None:
    if path := os.getenv("GITHUB_OUTPUT"):
        with open(path, "a") as fh:
            for key, value in outputs.items():
                fh.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as fh:
            fh.write(summary_md + "\n")


def _summary_markdown(report: dict[str, Any]) -> str:
    drift = report["drift"]
    lines = [
        "## Automated retraining report",
        "",
        f"- **Run at:** {report['run_at']}",
        f"- **Data:** {report['data_source']} ({report['new_rows']:,} rows)",
        f"- **Dataset drift:** {drift['dataset_drift']} "
        f"({drift['n_drifted_features']}/{drift['n_features']} features drifted)",
        f"- **Champion v{report['champion_version']} AUC on new data:** "
        f"{report['champion_metrics_new_data']['roc_auc']}",
        f"- **Retraining triggered:** {report['retrain_triggered']} "
        f"({', '.join(report['trigger_reasons']) or 'no trigger'})",
    ]
    if report.get("challenger_metrics"):
        lines.append(f"- **Challenger AUC:** {report['challenger_metrics']['roc_auc']}")
    lines.append(f"- **Decision:** {report['decision']}")
    lines += ["", "| Feature | PSI | Drifted |", "|---|---|---|"]
    for name, f in drift["features"].items():
        lines.append(f"| {name} | {f['psi']} | {'yes' if f['drifted'] else 'no'} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- pipeline
def run_pipeline(
    drift: float | None = None,
    data_path: Path | None = None,
    force: bool = False,
    n_samples: int = 8000,
    seed: int | None = None,
    model_dir: Path = MODEL_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    run_at = datetime.now(UTC)
    drift = drift_level_for(run_at.date()) if drift is None else drift
    seed = seed if seed is not None else int(run_at.timestamp()) % 100_000

    champion, champion_meta, reference = load_production_model(model_dir)
    log.info("Champion: v%s (AUC %s)", champion_meta["version"], champion_meta["metrics"]["roc_auc"])

    # 1-2. ingest + validate
    new_df, source = ingest(data_path, drift, n_samples, seed)
    problems = validate(new_df)
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    log.info("Ingested %s rows from %s", len(new_df), source)

    new_train, new_holdout = train_test_split(
        new_df, test_size=0.3, random_state=0, stratify=new_df[TARGET]
    )

    with tracking_run("retraining-pipeline", tags={"stage": "retraining"}) as run:
        # 3. monitor
        drift_report = detect_drift(reference[FEATURES], new_df[FEATURES])
        champion_new = evaluate(champion, new_holdout)
        auc_drop = round(champion_meta["metrics"]["roc_auc"] - champion_new["roc_auc"], 4)
        log.info(
            "Drift=%s (%s features) | champion AUC on new data %s (drop %s)",
            drift_report["dataset_drift"],
            drift_report["n_drifted_features"],
            champion_new["roc_auc"],
            auc_drop,
        )

        # 4. decide
        reasons: list[str] = []
        if force:
            reasons.append("forced")
        if drift_report["dataset_drift"]:
            reasons.append("data drift")
        if champion_new["roc_auc"] < MIN_ACCEPTABLE_AUC:
            reasons.append(f"AUC below {MIN_ACCEPTABLE_AUC}")
        if auc_drop > MAX_AUC_DROP:
            reasons.append(f"AUC dropped by {auc_drop}")

        run.log_params(
            {"simulated_drift": drift, "data_source": source, "new_rows": len(new_df), "seed": seed}
        )
        run.log_metrics(
            {
                "share_drifted_features": drift_report["share_drifted_features"],
                "champion_auc_drop": auc_drop,
            }
        )
        run.log_metrics(champion_new, prefix="champion_")
        run.log_dict(drift_report, "drift_report.json")

        report: dict[str, Any] = {
            "run_at": run_at.isoformat(timespec="seconds"),
            "data_source": source,
            "simulated_drift": drift,
            "new_rows": len(new_df),
            "champion_version": champion_meta["version"],
            "champion_metrics_new_data": champion_new,
            "champion_auc_drop": auc_drop,
            "drift": drift_report,
            "retrain_triggered": bool(reasons),
            "trigger_reasons": reasons,
            "challenger_metrics": None,
            "promoted": False,
            "new_version": None,
            "mlflow_run_id": run.run_id,
        }

        if not reasons:
            report["decision"] = "no retraining needed - champion kept"
        else:
            # 5. train challenger on recent data + slice of older data
            train_df = pd.concat([new_train, reference.sample(frac=0.5, random_state=1)])
            challenger = train(train_df, champion_meta.get("params"))
            # 6. compare on the same fresh hold-out
            challenger_new = evaluate(challenger, new_holdout)
            report["challenger_metrics"] = challenger_new
            run.log_metrics(challenger_new, prefix="challenger_")

            # 7. promote only if better
            if challenger_new["roc_auc"] >= champion_new["roc_auc"] + MIN_IMPROVEMENT:
                reason = (
                    f"{', '.join(reasons)}; AUC {champion_new['roc_auc']} -> "
                    f"{challenger_new['roc_auc']}"
                )
                run.log_model(
                    challenger, input_example=new_train[FEATURES].head(3), register=True
                )
                meta = save_production_model(
                    challenger,
                    challenger_new,
                    reference=train_df,
                    params=champion_meta.get("params"),
                    model_dir=model_dir,
                    extra={
                        "mlflow_run_id": run.run_id,
                        "training_rows": len(train_df),
                        "data_source": source,
                        "promotion_reason": reason,
                        "previous_version": champion_meta["version"],
                    },
                )
                report.update(
                    promoted=True,
                    new_version=meta["version"],
                    decision=f"challenger promoted to v{meta['version']}",
                )
            else:
                report["decision"] = "challenger not better - champion kept"
        run.set_tags({"decision": report["decision"], "promoted": report["promoted"]})

    # reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "latest_retrain_report.json").write_text(json.dumps(report, indent=2))
    with open(reports_dir / "retrain_history.jsonl", "a") as fh:
        slim = {k: v for k, v in report.items() if k != "drift"}
        slim["share_drifted_features"] = drift_report["share_drifted_features"]
        fh.write(json.dumps(slim) + "\n")

    _write_github_outputs(
        {"promoted": report["promoted"], "new_version": report["new_version"] or ""},
        _summary_markdown(report),
    )
    log.info("Decision: %s", report["decision"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift-aware automated retraining pipeline")
    parser.add_argument("--drift", type=float, default=None, help="simulated drift 0..1")
    parser.add_argument("--data", type=Path, default=None, help="CSV of new labelled data")
    parser.add_argument("--force", action="store_true", help="retrain even without triggers")
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(
        drift=args.drift,
        data_path=args.data,
        force=args.force,
        n_samples=args.n_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
