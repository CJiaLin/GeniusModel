from datetime import datetime
from pydantic import BaseModel, Field

from app.schemas.common import Metric, RunStatus


class RunCreate(BaseModel):
    project_id: str
    optimize_target: str = Field(default="AUC")
    generations: int = Field(default=20, ge=1, le=200)
    budget_minutes: int = Field(default=30, ge=1, le=600)


class Run(BaseModel):
    id: str
    project_id: str
    status: RunStatus
    logs: list[str]
    metrics: list[Metric]
    started_at: datetime | None = None
    ended_at: datetime | None = None


class RunReport(BaseModel):
    run_id: str
    runtime_minutes: float
    technical_summary: dict[str, float]
    business_summary: dict[str, float]
    iteration_advice: list[str]
    top_features: list[str]
