from datetime import datetime
from pydantic import BaseModel, Field

from app.schemas.common import TaskType


class ProjectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    scene: str = Field(description="fraud/marketing/custom")
    task_type: TaskType
    label_column: str
    description: str | None = None


class Project(BaseModel):
    id: str
    name: str
    scene: str
    task_type: TaskType
    label_column: str
    description: str | None = None
    created_at: datetime
