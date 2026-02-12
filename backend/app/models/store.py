from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import datetime

from app.schemas.common import Metric, RunStatus, TaskType


@dataclass
class ProjectRecord:
    id: str
    name: str
    scene: str
    task_type: TaskType
    label_column: str
    description: str | None
    created_at: datetime


@dataclass
class RunRecord:
    id: str
    project_id: str
    status: RunStatus
    logs: list[str] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class InMemoryStore:
    def __init__(self) -> None:
        self.projects: MutableMapping[str, ProjectRecord] = {}
        self.runs: MutableMapping[str, RunRecord] = {}


store = InMemoryStore()
