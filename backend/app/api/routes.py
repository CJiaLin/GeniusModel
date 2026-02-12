from fastapi import APIRouter

from app.schemas.project import Project, ProjectCreate
from app.schemas.run import Run, RunCreate, RunReport
from app.services.project_service import project_service
from app.services.run_service import run_service

router = APIRouter(prefix="/api/v1")


@router.post("/projects", response_model=Project)
def create_project(req: ProjectCreate) -> Project:
    return project_service.create(req)


@router.get("/projects", response_model=list[Project])
def list_projects() -> list[Project]:
    return project_service.list()


@router.post("/runs", response_model=Run)
def create_run(req: RunCreate) -> Run:
    return run_service.create_and_execute(req)


@router.get("/runs/{run_id}", response_model=Run)
def get_run(run_id: str) -> Run:
    return run_service.get(run_id)


@router.get("/runs/{run_id}/report", response_model=RunReport)
def get_run_report(run_id: str) -> RunReport:
    return run_service.build_report(run_id)
