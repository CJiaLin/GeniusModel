from enum import Enum
from pydantic import BaseModel


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"
    timeseries = "timeseries"


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Metric(BaseModel):
    name: str
    value: float
