from datetime import datetime
from uuid import uuid4

from app.models.store import ProjectRecord, store
from app.schemas.project import Project, ProjectCreate


class ProjectService:
    def create(self, req: ProjectCreate) -> Project:
        pid = f"prj_{uuid4().hex[:10]}"
        record = ProjectRecord(
            id=pid,
            name=req.name,
            scene=req.scene,
            task_type=req.task_type,
            label_column=req.label_column,
            description=req.description,
            created_at=datetime.utcnow(),
        )
        store.projects[pid] = record
        return Project(**record.__dict__)

    def list(self) -> list[Project]:
        return [Project(**p.__dict__) for p in store.projects.values()]


project_service = ProjectService()
