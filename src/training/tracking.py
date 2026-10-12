"""Thin wrapper around MLflow experiment tracking.

* If ``MLFLOW_TRACKING_URI`` is set (e.g. a DagsHub or self-hosted MLflow
  server) runs are logged there, including the model and a Model Registry
  entry.
* Otherwise runs are logged to a local ``./mlruns`` folder; view them with
  ``mlflow ui``.
* If MLflow is not installed or the server cannot be reached, training still
  works: tracking failures are logged as warnings and never break the
  pipeline.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from src.config import MLFLOW_EXPERIMENT, REGISTERED_MODEL_NAME, ROOT_DIR

log = logging.getLogger(__name__)


class _NullRun:
    run_id: str | None = None

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None: ...
    def set_tags(self, tags: dict[str, Any]) -> None: ...
    def log_dict(self, data: dict, name: str) -> None: ...
    def log_model(self, model: Any, input_example: Any = None, register: bool = False) -> None: ...


class _MlflowRun(_NullRun):
    def __init__(self, mlflow_module: Any, run: Any) -> None:
        self._mlflow = mlflow_module
        self.run_id = run.info.run_id

    def _safe(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # tracking must never break training
            log.warning("MLflow logging failed: %s", exc)

    def log_params(self, params: dict[str, Any]) -> None:
        self._safe(self._mlflow.log_params, {k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        clean = {f"{prefix}{k}": float(v) for k, v in metrics.items() if v is not None}
        self._safe(self._mlflow.log_metrics, clean)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._safe(self._mlflow.set_tags, {k: str(v) for k, v in tags.items()})

    def log_dict(self, data: dict, name: str) -> None:
        self._safe(self._mlflow.log_dict, data, name)

    def log_model(self, model: Any, input_example: Any = None, register: bool = False) -> None:
        def _log() -> None:
            import mlflow.sklearn

            kwargs: dict[str, Any] = {
                "input_example": input_example,
                "serialization_format": "cloudpickle",
            }
            if register:
                kwargs["registered_model_name"] = REGISTERED_MODEL_NAME
            try:
                mlflow.sklearn.log_model(model, name="model", **kwargs)
            except TypeError:  # MLflow 2.x uses artifact_path
                mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)

        self._safe(_log)


@contextmanager
def tracking_run(run_name: str, tags: dict[str, Any] | None = None) -> Iterator[_NullRun]:
    if os.getenv("DISABLE_MLFLOW", "").lower() in {"1", "true", "yes"}:
        yield _NullRun()
        return
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow not installed - skipping experiment tracking")
        yield _NullRun()
        return

    try:
        if not os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        active = mlflow.start_run(run_name=run_name)
    except Exception as exc:
        log.warning("Could not start MLflow run (%s) - continuing without tracking", exc)
        yield _NullRun()
        return

    run = _MlflowRun(mlflow, active)
    run.set_tags({"git_sha": os.getenv("GITHUB_SHA", "local"), **(tags or {})})
    try:
        yield run
    finally:
        try:
            mlflow.end_run()
        except Exception as exc:
            log.warning("Could not end MLflow run: %s", exc)


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db (local)")
