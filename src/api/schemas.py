"""Request / response schemas for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 34,
                "tenure_months": 5,
                "monthly_charges": 89.5,
                "total_charges": 447.5,
                "num_support_calls": 4,
                "avg_daily_usage_gb": 6.2,
                "contract_type": "month_to_month",
                "internet_service": "fiber",
                "payment_method": "electronic_check",
            }
        }
    )

    age: int = Field(..., ge=18, le=100)
    tenure_months: int = Field(..., ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float = Field(..., ge=0)
    num_support_calls: int = Field(..., ge=0, le=50)
    avg_daily_usage_gb: float = Field(..., ge=0, le=200)
    contract_type: Literal["month_to_month", "one_year", "two_year"]
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal["credit_card", "bank_transfer", "electronic_check"]


class PredictionResponse(BaseModel):
    prediction_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: Literal["low", "medium", "high"]
    model_version: int
    latency_ms: float


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    model_version: int
    count: int
    latency_ms: float
    predictions: list[PredictionResponse]
