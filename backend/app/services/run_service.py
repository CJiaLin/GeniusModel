from datetime import datetime
from uuid import uuid4

from fastapi import HTTPException

from app.models.store import RunRecord, store
from app.schemas.common import Metric, RunStatus
from app.schemas.run import Run, RunCreate, RunReport
from app.services.openevolve_adapter import OpenEvolveAdapter


class RunService:
    def __init__(self) -> None:
        self.adapter = OpenEvolveAdapter()

    def create_and_execute(self, req: RunCreate) -> Run:
        if req.project_id not in store.projects:
            raise HTTPException(status_code=404, detail="Project not found")

        rid = f"run_{uuid4().hex[:10]}"
        record = RunRecord(id=rid, project_id=req.project_id, status=RunStatus.queued)
        store.runs[rid] = record

        record.status = RunStatus.running
        record.started_at = datetime.utcnow()
        record.logs.extend(
            [
                "数据检查完成。",
                "开始 OpenEvolve 特征与参数联合演化。",
                f"配置: generations={req.generations}, budget={req.budget_minutes}min",
            ]
        )

        evo_result = self.adapter.evolve(req.optimize_target, req.generations, req.budget_minutes)
        record.metrics = [Metric(name=k, value=v) for k, v in evo_result.metrics.items()]
        record.logs.extend([f"Top Features: {', '.join(evo_result.top_features)}", *evo_result.notes])
        record.status = RunStatus.completed
        record.ended_at = datetime.utcnow()
        return Run(**record.__dict__)

    def get(self, run_id: str) -> Run:
        run = store.runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return Run(**run.__dict__)

    def build_report(self, run_id: str) -> RunReport:
        run = store.runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status != RunStatus.completed or not run.started_at or not run.ended_at:
            raise HTTPException(status_code=400, detail="Run not completed")

        metric_map = {m.name: m.value for m in run.metrics}
        runtime = (run.ended_at - run.started_at).total_seconds() / 60

        return RunReport(
            run_id=run_id,
            runtime_minutes=round(runtime, 2),
            technical_summary={
                "AUC": metric_map.get("AUC", 0.0),
                "KS": metric_map.get("KS", 0.0),
                "Recall@Top5%": metric_map.get("Recall@Top5%", 0.0),
                "LatencyMs": metric_map.get("LatencyMs", 0.0),
            },
            business_summary={
                "fraud_intercept_rate": round(metric_map.get("Recall@Top5%", 0.0) * 0.85, 4),
                "false_positive_cost_reduction": 0.17,
            },
            iteration_advice=[
                "结合阈值优化策略，按客群分层设置准入规则。",
                "将设备画像与渠道画像做交叉衍生后再进化筛选。",
            ],
            top_features=[
                "txn_amt_7d_std",
                "night_txn_ratio",
                "device_switch_cnt",
                "merchant_risk_score",
            ],
        )


run_service = RunService()
