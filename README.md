# GeniusModel

基于 Python 的前后端分离自动化建模项目（冷启动建模 + OpenEvolve 演化 + 自动报告）。

## 目录结构

```text
GeniusModel/
├── backend/                     # FastAPI 后端
│   ├── app/
│   │   ├── api/routes.py        # REST API
│   │   ├── services/            # 项目、运行、OpenEvolve 适配
│   │   ├── schemas/             # Pydantic 模型
│   │   └── main.py              # 应用入口
│   ├── tests/                   # 后端 API 测试
│   └── requirements.txt
├── frontend/                    # 独立前端
│   ├── index.html
│   └── src/
├── openevolve/                  # OpenEvolve 预留（可直接替换成你的真实代码）
│   └── engine.py
└── README.md
```

## 已实现能力

- 项目创建（分类/回归/时间序列）
- 一键发起演化运行
- 自动生成报告（技术指标 + 业务摘要 + 优化建议）
- 前后端分离（REST API 调用）

## OpenEvolve 接入

默认后端导入：

```python
from openevolve.engine import OpenEvolveEngine
```

你可以直接把你的 OpenEvolve 代码覆盖 `openevolve/` 目录，保持 `OpenEvolveEngine.evolve(...)` 接口兼容即可。

## 启动

### 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger: `http://127.0.0.1:8000/docs`

### 前端

```bash
cd frontend
python -m http.server 5173
```

访问：`http://127.0.0.1:5173`

## API

- `POST /api/v1/projects`
- `GET /api/v1/projects`
- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs/{run_id}/report`

## 测试

```bash
cd backend
pytest
```
