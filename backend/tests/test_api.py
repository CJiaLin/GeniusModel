from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_project_run_report_flow() -> None:
    project_resp = client.post(
        '/api/v1/projects',
        json={
            'name': 'Fraud Demo',
            'scene': 'fraud',
            'task_type': 'classification',
            'label_column': 'is_fraud',
            'description': 'demo',
        },
    )
    assert project_resp.status_code == 200
    project = project_resp.json()

    run_resp = client.post(
        '/api/v1/runs',
        json={
            'project_id': project['id'],
            'optimize_target': 'AUC',
            'generations': 12,
            'budget_minutes': 20,
        },
    )
    assert run_resp.status_code == 200
    run = run_resp.json()
    assert run['status'] == 'completed'

    report_resp = client.get(f"/api/v1/runs/{run['id']}/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report['run_id'] == run['id']
    assert 'AUC' in report['technical_summary']
