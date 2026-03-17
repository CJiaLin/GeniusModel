"""
对话式AutoML的API服务模块

提供SSE流式输出和会话管理功能，支持：
1. 用户修改特征工程/建模思路
2. 保存模型文件
3. 生成完整可执行代码
4. 生成可视化报告
5. LLM驱动的意图识别
"""

import os
import sys
import logging
import json
import uuid
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AutoML-ChatAPI")
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np

from llm_client import get_llm_client, configure_llm

# 从配置文件加载LLM配置
from llm_client import load_config_from_file
_file_config = load_config_from_file()

if not _file_config.get("api_key"):
    raise ValueError("请在 config.yaml 中配置 LLM API Key")

configure_llm(
    base_url=_file_config.get("base_url", "https://fast.poloai.top"),
    api_key=_file_config.get("api_key"),
    model=_file_config.get("model", "claude-sonnet-4-20250514-thinking")
)

app = FastAPI(title="AutoML Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "data/models"
REPORT_DIR = "data/reports"
CODE_DIR = "data/codes"
DATA_DIR = "data/processed"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = "claude-sonnet-4-20250514-thinking"


class Session(BaseModel):
    id: str
    title: str
    created_at: str


class SessionData:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.messages: Dict[str, List[Dict]] = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self.data_raw: Dict[str, pd.DataFrame] = {}
        self.target_column: Dict[str, str] = {}
        self.current_stage: Dict[str, str] = {}
        self.current_thinking: Dict[str, str] = {}
        self.model: Dict[str, any] = {}
        self.model_path: Dict[str, str] = {}
        self.code_path: Dict[str, str] = {}
        self.report_path: Dict[str, str] = {}
        self.data_cleaned_path: Dict[str, str] = {}
        self.data_featured_path: Dict[str, str] = {}
        self.data_raw_path: Dict[str, str] = {}
        self.full_code: Dict[str, str] = {}
        self.llm = None
        
session_manager = SessionData()


@app.post("/sessions")
async def create_session():
    session_id = str(uuid.uuid4())
    session = Session(id=session_id, title="新会话", created_at=datetime.now().isoformat())
    session_manager.sessions[session_id] = session
    session_manager.messages[session_id] = []
    session_manager.current_stage[session_id] = None
    session_manager.current_thinking[session_id] = {}
    return session


@app.get("/sessions")
async def get_sessions():
    return list(session_manager.sessions.values())


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
    if session_id in session_manager.messages:
        del session_manager.messages[session_id]
    if session_id in session_manager.data:
        del session_manager.data[session_id]
    if session_id in session_manager.data_raw:
        del session_manager.data_raw[session_id]
    if session_id in session_manager.target_column:
        del session_manager.target_column[session_id]
    if session_id in session_manager.current_stage:
        del session_manager.current_stage[session_id]
    if session_id in session_manager.model_path:
        try:
            if os.path.exists(session_manager.model_path[session_id]):
                os.remove(session_manager.model_path[session_id])
        except:
            pass
        del session_manager.model_path[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.post("/data/upload")
async def upload_data(
    session_id: str = Form(...),
    target_column: str = Form(...),
    file: UploadFile = File(...)
):
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        if suffix == '.csv':
            df = pd.read_csv(tmp_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        session_manager.data[session_id] = df.copy()
        session_manager.data_raw[session_id] = df.copy()
        session_manager.target_column[session_id] = target_column
        
        # 保存原始数据文件
        raw_path = f"{DATA_DIR}/data_raw_{session_id}.csv"
        df.to_csv(raw_path, index=False)
        session_manager.data_raw_path[session_id] = raw_path
        
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df.select_dtypes(include=['object']).columns)
        
        return {
            "success": True,
            "message": f"数据加载成功: {df.shape[0]}行 × {df.shape[1]}列",
            "download_url": f"/data/download/raw/{session_id}",
            "profile": {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "missing": df.isnull().sum().to_dict(),
                "target_column": target_column
            }
        }
    finally:
        os.unlink(tmp_path)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    import asyncio
    from fastapi.responses import StreamingResponse
    
    session_id = request.session_id
    logger.info(f"[Chat] 收到消息 from session={session_id}: {request.message[:50]}...")
    
    if session_id not in session_manager.messages:
        session_manager.messages[session_id] = []
    
    messages = session_manager.messages[session_id]
    messages.append({"role": "user", "content": request.message})
    
    async def generate():
        try:
            logger.info(f"[Chat] 开始处理请求, session={session_id}")
            
            yield f"data: {json.dumps({'type': 'step', 'name': '理解问题', 'status': 'running'})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'type': 'step', 'name': '处理请求', 'status': 'running'})}\n\n"
            await asyncio.sleep(0.1)
            
            logger.info(f"[Chat] 调用 generate_response, session={session_id}")
            response_content, stage_update, thinking_update = await generate_response(session_id, request.message, messages)
            logger.info(f"[Chat] generate_response 完成, response长度={len(response_content)}")
            
            if stage_update:
                session_manager.current_stage[session_id] = stage_update
            if thinking_update:
                session_manager.current_thinking[session_id].update(thinking_update)
            
            yield f"data: {json.dumps({'type': 'step', 'name': '生成回复', 'status': 'running'})}\n\n"
            
            for chunk in split_into_chunks(response_content, 20):
                yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
                await asyncio.sleep(0.01)
            
            messages.append({"role": "assistant", "content": response_content})
            
            yield f"data: {json.dumps({'type': 'step', 'name': '完成', 'status': 'completed'})}\n\n"
            logger.info(f"[Chat] 处理完成, session={session_id}")
            
        except Exception as e:
            logger.error(f"[Chat] 处理错误: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


def split_into_chunks(text: str, chunk_size: int = 20) -> list:
    """将文本分割成指定大小的块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def is_confirm(message: str) -> bool:
    confirm_keywords = ["确认", "执行", "是", "好的", "开始", "run", "execute", "yes", "ok"]
    return any(kw in message.lower() for kw in confirm_keywords)


def is_modify(message: str) -> bool:
    modify_keywords = ["修改", "调整", "改变", "不要", "不用", "去掉", "添加", "增加", "减少", "改"]
    return any(kw in message.lower() for kw in modify_keywords)


async def llm_intent_recognition(
    session_id: str, 
    message: str, 
    current_stage: str = None,
    thinking: dict = None,
    messages: List[Dict] = None
) -> dict:
    """
    使用LLM进行意图识别，返回结构化的意图和参数
    """
    logger.info(f"[Intent] 开始意图识别, session={session_id}")
    
    try:
        llm = get_llm_client(temperature=0.1)
        logger.info(f"[Intent] LLM客户端创建成功, model={llm.model_name}")
    except Exception as e:
        logger.error(f"[Intent] LLM客户端创建失败: {str(e)}", exc_info=True)
        return {
            "intent": "chat",
            "confidence": 0.5,
            "reasoning": f"LLM初始化失败: {str(e)}",
            "params": {}
        }
    
    has_data = session_id in session_manager.data
    has_model = session_manager.model.get(session_id) is not None
    data_shape = session_manager.data.get(session_id).shape if has_data else None
    target = session_manager.target_column.get(session_id)
    
    system_prompt = f"""你是一个AutoML对话系统的意图识别助手。你的任务是根据用户的消息和当前对话状态，理解用户的意图。

## 当前对话状态
- 是否有数据: {has_data}
- 数据形状: {data_shape}
- 目标列: {target}
- 是否有训练好的模型: {has_model}
- 当前阶段: {current_stage or "无"}
- 待确认的思路: {thinking}

## 可识别的意图类型

### 1. confirm (确认执行)
当用户确认执行当前提出的方案时触发，如：
- "确认"、"执行"、"是"、"好的"、"开始"
- "运行吧"、"开始吧"、"开始执行"

### 2. modify (修改思路)
当用户想要修改当前方案时触发，如：
- "修改"、"调整"、"改变"
- "不要这个"、"换成"、"换一个"
- "添加XX"、"去掉XX"

### 3. analysis (数据分析)
当用户想要进行数据分析时触发，如：
- "分析数据"、"数据分析"、"查看数据"
- "数据概况"、"统计信息"、"可视化"

### 4. cleaning (数据清洗)
当用户想要进行数据清洗时触发，如：
- "清洗"、"数据清洗"、"处理缺失值"
- "预处理"、"清理数据"

### 5. feature (特征工程)
当用户想要进行特征工程时触发，如：
- "特征"、"特征工程"、"创建特征"
- "提取特征"、"特征提取"

### 5. model (模型训练)
当用户想要训练模型时触发，如：
- "训练"、"训练模型"、"建模"
- "开始训练"

### 6. export_code (导出代码)
当用户想要获取完整代码时触发，如：
- "导出代码"、"下载代码"、"获取代码"
- "生成代码"、"代码"

### 7. generate_report (生成报告)
当用户想要生成分析报告时触发，如：
- "报告"、"分析报告"、"可视化"
- "生成报告"、"分析"

### 8. download_model (下载模型)
当用户想要下载模型时触发，如：
- "下载模型"、"模型下载"
- "获取模型"

### 9. download_data (下载数据)
当用户想要下载数据时触发，如：
- "下载数据"、"导出数据"
- "获取数据"

### 10. chat (闲聊/其他)
当用户想要聊天或问题解答时触发

## 输出格式要求

请以JSON格式输出你的分析结果：
```json
{{
    "intent": "意图类型",
    "confidence": 0.95,
    "reasoning": "简短的分析理由",
    "params": {{"参数": "值"}},
    "response_hint": "给后续处理的简短提示"
}}
```

注意：
- intent必须是上述意图类型之一
- confidence是0-1之间的置信度
- 如果多个意图都可能，选择最匹配的那个
- 如果当前有待确认的阶段(cleaning/feature/model)，优先判断是否在确认或修改
"""

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户消息: {message}\n\n请分析用户的意图。"}
        ])
        
        result_text = response.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        intent_data = json.loads(result_text.strip())
        return intent_data
        
    except Exception as e:
        print(f"LLM意图识别失败: {e}")
        return {
            "intent": "chat",
            "confidence": 0.5,
            "reasoning": "意图识别失败，回退到闲聊模式",
            "params": {},
            "response_hint": ""
        }


async def generate_response(session_id: str, message: str, messages: List[Dict]) -> tuple:
    message_lower = message.lower()
    current_stage = session_manager.current_stage.get(session_id)
    thinking = session_manager.current_thinking.get(session_id, {})
    
    intent_data = await llm_intent_recognition(
        session_id, message, current_stage, thinking, messages
    )
    
    intent = intent_data.get("intent", "chat")
    confidence = intent_data.get("confidence", 0.5)
    print(f"[LLM意图识别] intent={intent}, confidence={confidence}, reasoning={intent_data.get('reasoning', '')}")
    
    if session_id in session_manager.data:
        df = session_manager.data[session_id]
        target = session_manager.target_column.get(session_id, "未知")
        
        if current_stage and intent == "modify":
            if current_stage == "cleaning":
                modified_thinking = modify_cleaning_thinking(message, thinking.get("cleaning", ""))
                session_manager.current_thinking[session_id]["cleaning"] = modified_thinking
                return f"""## 🔍 修改后的数据清洗思路

{modified_thinking}

**请回复"确认"或"执行"开始清洗数据**
""", "cleaning", {"cleaning": modified_thinking}
            
            elif current_stage == "feature":
                modified_thinking = modify_feature_thinking(message, thinking.get("feature", ""))
                session_manager.current_thinking[session_id]["feature"] = modified_thinking
                return f"""## 🔧 修改后的特征工程思路

{modified_thinking}

**请回复"确认"或"执行"开始特征工程**
""", "feature", {"feature": modified_thinking}
            
            elif current_stage == "model":
                modified_thinking = modify_model_thinking(message, thinking.get("model", ""))
                session_manager.current_thinking[session_id]["model"] = modified_thinking
                return f"""## 🤖 修改后的建模思路

{modified_thinking}

**请回复"确认"或"执行"开始训练模型**
""", "model", {"model": modified_thinking}
        
        # 处理确认执行
        if current_stage and intent == "confirm":
            stage = current_stage
            session_manager.current_stage[session_id] = None
            
            if stage == "cleaning":
                custom_thinking = thinking.get("cleaning")
                result = execute_cleaning(df, custom_thinking)
                session_manager.data[session_id] = result["data"]
                
                # 保存清洗后的数据
                cleaned_path = f"{DATA_DIR}/data_cleaned_{session_id}.csv"
                result["data"].to_csv(cleaned_path, index=False)
                session_manager.data_cleaned_path[session_id] = cleaned_path
                
                full_code = generate_full_pipeline_code(
                    session_manager.data_raw.get(session_id),
                    result["data"],
                    session_manager.target_column.get(session_id),
                    session_manager.model.get(session_id)
                )
                session_manager.full_code[session_id] = full_code
                
                # 同时保存代码文件
                code_path = f"{CODE_DIR}/pipeline_{session_id}.py"
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(full_code)
                session_manager.code_path[session_id] = code_path
                
                return f"""## ✅ 数据清洗完成！

{result['message']}

处理后数据：
- 形状: {result['data'].shape[0]}行 × {result['data'].shape[1]}列
- 缺失值数量: {result['data'].isnull().sum().sum()}

{result.get('output', '')}

### 📥 数据下载
清洗后的数据已保存，您可以直接下载：
- `/data/download/cleaned/{session_id}`

继续选择下一步操作：
- "特征工程" - 创建新特征
- "训练模型" - 训练模型
- "修改思路" - 调整清洗方案
- "导出代码" - 获取完整建模代码
- "生成报告" - 生成可视化分析报告
""", None, None
            
            elif stage == "feature":
                custom_thinking = thinking.get("feature")
                result = execute_feature_engineering(df, custom_thinking)
                session_manager.data[session_id] = result["data"]
                
                # 保存特征工程后的数据
                featured_path = f"{DATA_DIR}/data_featured_{session_id}.csv"
                result["data"].to_csv(featured_path, index=False)
                session_manager.data_featured_path[session_id] = featured_path
                
                full_code = generate_full_pipeline_code(
                    session_manager.data_raw.get(session_id),
                    result["data"],
                    session_manager.target_column.get(session_id),
                    session_manager.model.get(session_id)
                )
                session_manager.full_code[session_id] = full_code
                
                # 同时保存代码文件
                code_path = f"{CODE_DIR}/pipeline_{session_id}.py"
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(full_code)
                session_manager.code_path[session_id] = code_path
                
                return f"""## ✅ 特征工程完成！

{result['message']}

处理后数据：
- 形状: {result['data'].shape[0]}行 × {result['data'].shape[1]}列

{result.get('output', '')}

### 📥 数据下载
特征工程后的数据已保存，您可以直接下载：
- `/data/download/featured/{session_id}`

继续选择下一步操作：
- "训练模型" - 训练模型
- "修改思路" - 调整特征方案
- "导出代码" - 获取完整建模代码
- "生成报告" - 生成可视化分析报告
""", None, None
            
            elif stage == "model":
                result, model = execute_model_training_with_model(df, target)
                session_manager.model[session_id] = model
                
                model_path = f"{MODEL_DIR}/model_{session_id}.joblib"
                import joblib
                joblib.dump(model, model_path)
                session_manager.model_path[session_id] = model_path
                
                full_code = generate_full_pipeline_code(
                    session_manager.data_raw.get(session_id),
                    session_manager.data.get(session_id),
                    target,
                    model
                )
                session_manager.full_code[session_id] = full_code
                
                # 同时保存代码到文件
                code_path = f"{CODE_DIR}/pipeline_{session_id}.py"
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(full_code)
                session_manager.code_path[session_id] = code_path
                
                return f"""## ✅ 模型训练完成！

{result['message']}

{result.get('output', '')}

### 📦 模型文件
模型已保存至: `{model_path}`

### 📊 后续操作
- "导出代码" - 获取完整建模代码
- "生成报告" - 生成可视化分析报告
- "下载模型" - 下载训练好的模型文件
""", None, None
        
        # 阶段选择 - 基于LLM意图识别
        if intent in ["analysis", "cleaning", "feature", "model"]:
            if intent == "analysis":
                # 使用新的数据分析流程：思路 -> 代码 -> 执行
                yield f"data: {json.dumps({'type': 'step', 'name': '生成分析思路', 'status': 'running'})}\n\n"
                
                analysis_result = await perform_data_analysis(df, target)
                
                if analysis_result["success"]:
                    yield f"data: {json.dumps({'type': 'step', 'name': '执行分析代码', 'status': 'completed'})}\n\n"
                    
                    report = format_analysis_results(analysis_result, target)
                    
                    # 保存分析结果到session
                    session_manager.current_thinking[session_id]["analysis_result"] = analysis_result
                    session_manager.current_thinking[session_id]["analysis_code"] = analysis_result.get("code", "")
                    
                    # 流式返回报告
                    for chunk in split_into_chunks(report, 100):
                        yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
                    
                    return None, "analysis", {"analysis": analysis_result}
                else:
                    error_msg = f"数据分析失败: {analysis_result.get('error', '未知错误')}"
                    yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                    return error_msg, None, None
            
            elif intent == "cleaning":
                thinking_text = generate_cleaning_thinking(df)
                return f"""## 🔍 数据清洗思路

{thinking_text}

**请回复"确认"执行，或"修改"调整方案**
""", "cleaning", {"cleaning": thinking_text}
        
            elif intent == "feature":
                thinking_text = generate_feature_thinking(df)
                return f"""## 🔧 特征工程思路

{thinking_text}

**请回复"确认"执行，或"修改"调整方案**
""", "feature", {"feature": thinking_text}
        
            elif intent == "model":
                thinking_text = generate_model_thinking(df, target)
                return f"""## 🤖 建模思路

{thinking_text}

**请回复"确认"执行，或"修改"调整方案**
""", "model", {"model": thinking_text}
        
        # 处理导出代码、生成报告、下载等请求
        if intent == "export_code":
            code = session_manager.full_code.get(session_id, "# 暂无完整代码")
            code_path = f"{CODE_DIR}/pipeline_{session_id}.py"
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code)
            session_manager.code_path[session_id] = code_path
            
            return f"""## 📝 完整建模代码

```python
{code}
```

---

### 📥 代码下载

请访问以下链接下载代码文件：

`/code/download/{session_id}`
""", None, None
        
        if intent == "generate_report":
            report = generate_analysis_report(session_id)
            report_path = f"{REPORT_DIR}/report_{session_id}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            session_manager.report_path[session_id] = report_path
            return f"""## 📊 分析报告

{report}

---

### 📥 报告下载
请访问以下链接下载报告：

`/report/download/{session_id}`
""", None, None
        
        if intent == "download_model":
            model_path = session_manager.model_path.get(session_id)
            if model_path and os.path.exists(model_path):
                return f"""## 📦 模型下载

请访问以下链接下载模型文件：

`/model/download/{session_id}`
""", None, None
        
        if intent == "download_data":
            return """### 📥 数据下载

您可以下载以下数据文件：
- 原始数据: `/data/download/raw/{session_id}`
- 清洗后数据: `/data/download/cleaned/{session_id}`
- 特征工程后数据: `/data/download/featured/{session_id}`
""", None, None
    
    # 闲聊模式 - 引导用户开始建模流程
    if intent == "chat" or confidence < 0.7:
        has_data = session_id in session_manager.data
        
        # 如果用户已经上传了数据，引导开始数据分析
        if has_data:
            df = session_manager.data[session_id]
            target = session_manager.target_column.get(session_id, "未知")
            
            # 分析用户消息是否包含建模目标
            llm = get_llm_client(temperature=0.3)
            analysis_prompt = f"""分析用户的建模需求，提取关键信息：

用户消息：{message}
数据信息：
- 形状：{df.shape}
- 列名：{list(df.columns)}
- 目标列：{target}

请分析：
1. 用户是否明确了建模目标？（是/否）
2. 任务类型是什么？（分类/回归/其他/未明确）
3. 用户是否需要引导开始建模流程？（是/否）

以JSON格式返回：
{{"has_goal": true/false, "task_type": "classification/regression/other/unknown", "needs_guidance": true/false}}"""
            
            try:
                analysis_response = await llm.ainvoke([
                    {"role": "system", "content": "你是一个专业的AutoML需求分析师。"},
                    {"role": "user", "content": analysis_prompt}
                ])
                
                analysis_text = analysis_response.content.strip()
                if "```json" in analysis_text:
                    analysis_text = analysis_text.split("```json")[1].split("```")[0]
                elif "```" in analysis_text:
                    analysis_text = analysis_text.split("```")[1].split("```")[0]
                
                import json
                analysis = json.loads(analysis_text.strip())
                
                # 如果用户需要引导，提供逐步建模引导
                if analysis.get("needs_guidance", True):
                    task_type = analysis.get("task_type", "unknown")
                    
                    return f"""## 🤖 欢迎使用 AutoML 智能建模助手！

我已收到您的建模需求，让我帮您一步步完成建模流程。

### 📊 当前数据状态
- **数据形状**: {df.shape[0]} 行 × {df.shape[1]} 列
- **目标列**: {target}
- **任务类型**: {task_type if task_type != "unknown" else "待确认"}

### 🚀 让我们开始建模流程

**第一步：数据分析**
在进行数据清洗之前，我们先对数据进行全面分析，了解数据质量和分布特征。

请回复 **"分析数据"** 或 **"数据分析"**，我将为您生成详细的数据分析报告（包含可视化图表）。

---

💡 **完整流程预览**：
1. � **数据分析** → 2. �🔍 数据清洗 → 3. 🔧 特征工程 → 4. 🤖 模型训练 → 5. 📊 结果评估

您也可以直接说：
- "查看数据" - 了解数据概况
- "分析目标列" - 分析目标变量分布
- "跳过分析直接清洗" - 如果您已经了解数据情况""", None, None
                    
            except Exception as e:
                logger.error(f"需求分析失败: {e}", exc_info=True)
        
        # 如果没有数据，引导上传数据
        else:
            return """## 👋 欢迎使用 AutoML 智能建模助手！

我是您的智能建模助手，将帮助您完成从数据分析到模型训练的完整流程。

### 📤 请先上传数据

请点击左侧的 **"上传数据"** 按钮，上传您的数据文件（CSV格式）。

上传后，请告诉我：
- 您的建模目标是什么？（例如：预测房价、识别欺诈交易等）
- 目标列是哪一列？

我将根据您的需求，一步步引导您完成建模流程。

---

💡 **我能帮您做什么**：
- 📊 **数据分析** - 全面分析数据质量，生成可视化报告
- 🔍 **数据清洗** - 自动检测并处理缺失值、异常值
- 🔧 **特征工程** - 传统特征 + LLM智能特征生成
- 🤖 **模型训练** - 自动选择最优模型
- 📊 **结果分析** - 可视化报告和模型评估
- 💾 **导出交付** - 完整代码和训练好的模型""", None, None
    
    return f"""收到您的消息：「{message}」

当前状态：{get_status_text(session_id)}

可以说：
- "分析数据" - 开始数据分析（推荐第一步）
- "清洗数据" - 开始数据清洗
- "特征工程" - 创建新特征
- "训练模型" - 训练模型
- "导出代码" - 获取完整代码
- "生成报告" - 生成分析报告""", None, None


def get_status_text(session_id: str) -> str:
    if session_id not in session_manager.data:
        return "未加载数据"
    df = session_manager.data[session_id]
    target = session_manager.target_column.get(session_id, "未知")
    return f"已加载数据: {df.shape[0]}行×{df.shape[1]}列, 目标列: {target}"


def generate_data_analysis(df: pd.DataFrame, target: str = None) -> dict:
    """
    生成全面的数据分析报告，参考 data-analysis skill 的最佳实践
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    
    analysis = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "numeric_summary": {},
        "categorical_summary": {},
        "visualizations": {}
    }
    
    # 数值列分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        analysis["numeric_summary"][col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "q25": df[col].quantile(0.25),
            "q75": df[col].quantile(0.75),
            "missing": df[col].isnull().sum()
        }
    
    # 类别列分析
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        analysis["categorical_summary"][col] = {
            "unique_count": df[col].nunique(),
            "top_values": value_counts.head(5).to_dict(),
            "missing": df[col].isnull().sum()
        }
    
    # 生成可视化 - 使用较低分辨率以减小体积
    try:
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 缺失值热力图
        if df.isnull().sum().sum() > 0:
            plt.figure(figsize=(8, 4), dpi=80)
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].head(15)  # 只显示前15个
            plt.bar(range(len(missing_data)), missing_data.values, color='steelblue')
            plt.xticks(range(len(missing_data)), missing_data.index, rotation=45, ha='right')
            plt.title('Missing Values by Column', fontsize=10)
            plt.ylabel('Count')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
            buffer.seek(0)
            analysis["visualizations"]["missing_values"] = base64.b64encode(buffer.read()).decode()
            plt.close()
        
        # 2. 数值特征分布 - 只显示前4个数值特征
        numeric_cols_to_plot = [c for c in numeric_cols if c != target][:4]
        if numeric_cols_to_plot:
            n_plots = len(numeric_cols_to_plot)
            rows = (n_plots + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(10, 3*rows), dpi=80)
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if rows > 1 else [axes]
            
            for i, col in enumerate(numeric_cols_to_plot):
                axes[i].hist(df[col].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
                axes[i].set_title(f'{col}', fontsize=9)
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')
            
            # 隐藏多余的子图
            for i in range(n_plots, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
            buffer.seek(0)
            analysis["visualizations"]["numeric_distributions"] = base64.b64encode(buffer.read()).decode()
            plt.close()
        
        # 3. 目标变量分析
        if target and target in df.columns:
            plt.figure(figsize=(6, 4), dpi=80)
            if df[target].dtype == 'object' or df[target].nunique() < 10:
                value_counts = df[target].value_counts().head(10)
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                plt.title(f'{target} Distribution', fontsize=10)
            else:
                plt.hist(df[target].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
                plt.title(f'{target} Distribution', fontsize=10)
                plt.xlabel(target)
                plt.ylabel('Frequency')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
            buffer.seek(0)
            analysis["visualizations"]["target_distribution"] = base64.b64encode(buffer.read()).decode()
            plt.close()
        
        # 4. 相关性热力图 - 只显示前10个数值特征
        numeric_cols_for_corr = [c for c in numeric_cols if c != target][:10]
        if len(numeric_cols_for_corr) > 1:
            plt.figure(figsize=(8, 6), dpi=80)
            corr_matrix = df[numeric_cols_for_corr].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.1f', cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix', fontsize=10)
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
            buffer.seek(0)
            analysis["visualizations"]["correlation_matrix"] = base64.b64encode(buffer.read()).decode()
            plt.close()
    
    except Exception as e:
        logger.error(f"生成可视化时出错: {e}", exc_info=True)
    
    return analysis



# 简化版本的函数定义

def format_analysis_report(analysis: dict, target: str = None) -> str:
    """格式化数据分析报告"""
    report = f"""# 📊 数据分析报告

## 1. 数据概览
| 指标 | 值 |
|------|-----|
| 样本数 | {analysis['shape'][0]} |
| 特征数 | {analysis['shape'][1]} |
| 目标列 | {target or '未指定'} |

## 2. 数据质量评估
"""
    
    missing_cols = {k: v for k, v in analysis['missing'].items() if v > 0}
    if missing_cols:
        report += "\n### 缺失值情况\n| 列名 | 缺失数 | 缺失率 |\n|------|--------|--------|\n"
        for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = analysis['missing_pct'][col]
            report += f"| {col} | {count} | {pct:.1f}% |\n"
    else:
        report += "\n✅ 数据完整，无缺失值\n"
    
    # 添加可视化
    report += "\n## 3. 数据可视化\n"
    if "missing_values" in analysis['visualizations']:
        report += f"\n### 缺失值分布\n![Missing Values](data:image/png;base64,{analysis['visualizations']['missing_values']})\n"
        report += "\n**图表解读**: 展示了各特征的缺失值数量，高缺失率特征建议删除或特殊处理\n"
    
    if "numeric_distributions" in analysis['visualizations']:
        report += f"\n### 数值特征分布\n![Numeric Distributions](data:image/png;base64,{analysis['visualizations']['numeric_distributions']})\n"
        report += "\n**图表解读**: 展示数值特征的分布情况，右偏分布建议对数变换\n"
    
    if "correlation_matrix" in analysis['visualizations']:
        report += f"\n### 特征相关性矩阵\n![Correlation Matrix](data:image/png;base64,{analysis['visualizations']['correlation_matrix']})\n"
        report += "\n**图表解读**: 展示特征间相关性，高相关性(>0.8)特征考虑删除\n"
    
    report += """\n## 4. 下一步建议
- 请回复 "确认分析结果" 生成数据清洗方案
- 或提出您的具体需求

---

**请回复 "确认分析结果" 继续**
"""
    return report


async def generate_cleaning_thinking(df: pd.DataFrame, target: str = None) -> str:
    """使用LLM动态生成数据清洗方案"""
    import json
    
    # 准备数据摘要
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    column_stats = []
    for col in df.columns[:15]:
        col_info = {"name": col, "dtype": str(df[col].dtype)}
        col_info["missing"] = int(df[col].isnull().sum())
        col_info["missing_pct"] = round(float(df[col].isnull().sum() / len(df) * 100), 2)
        
        if col in numeric_cols:
            col_info["type"] = "numeric"
            col_info["skew"] = round(float(df[col].skew()), 2)
        else:
            col_info["type"] = "categorical"
            col_info["unique"] = int(df[col].nunique())
        column_stats.append(col_info)
    
    prompt = f"""作为数据工程师，根据以下数据摘要生成个性化清洗方案：

数据: {len(df)}行 × {len(df.columns)}列, 目标: {target or '未指定'}

列信息: {json.dumps(column_stats, ensure_ascii=False)}

参考ML Engineering最佳实践：
1. 缺失值: >50%删除, 10-50%策略填充, <10%简单填充
2. 数值: 偏态用中位数, 正态用均值; 右偏>1.5用log1p变换
3. 异常值: IQR检测, Winsorization处理
4. 类别: 高基数(>20)目标编码, 低基数One-hot

请生成具体清洗方案，包括每列的处理方法。"""

    try:
        llm = get_llm_client(temperature=0.3)
        response = await llm.ainvoke([
            {"role": "system", "content": "你是数据工程专家，精通ML Engineering最佳实践。"},
            {"role": "user", "content": prompt}
        ])
        
        return f"""## 🔍 数据清洗方案\n\n{response.content.strip()}\n\n---\n\n**请回复 "确认" 执行，或提出修改意见**"""
    except Exception as e:
        return f"生成方案失败: {str(e)}"


def modify_cleaning_thinking(user_input: str, current_thinking: str) -> str:
    """修改清洗方案"""
    return f"""根据您的要求 "{user_input}" 调整方案：

原方案已更新，请查看修改后的方案。

---

**请回复 "确认" 执行修改后的方案**"""


async def generate_feature_thinking(df: pd.DataFrame, target: str = None) -> str:
    """使用LLM动态生成特征工程方案"""
    import json
    
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 分析特征
    feature_info = []
    for col in numeric_cols[:10]:
        feature_info.append({
            "name": col,
            "type": "numeric",
            "skew": round(float(df[col].skew()), 2)
        })
    for col in categorical_cols[:5]:
        feature_info.append({
            "name": col,
            "type": "categorical",
            "unique": int(df[col].nunique())
        })
    
    prompt = f"""作为特征工程专家，根据以下特征信息生成方案：

特征: {json.dumps(feature_info, ensure_ascii=False)}
目标: {target or '未指定'}

参考ML Engineering：
1. 数值变换: 右偏用log1p, 正态标准化
2. 交互特征: 比率、差值、多项式
3. 时间特征: 滞后、滚动统计、循环编码
4. 类别编码: 低基数One-hot, 高基数目标编码
5. 文本: TF-IDF, Embedding

请生成具体特征工程方案。"""

    try:
        llm = get_llm_client(temperature=0.3)
        response = await llm.ainvoke([
            {"role": "system", "content": "你是特征工程专家。"},
            {"role": "user", "content": prompt}
        ])
        
        return f"""## 🔧 特征工程方案\n\n{response.content.strip()}\n\n---\n\n**请回复 "确认" 执行，或提出修改意见**"""
    except Exception as e:
        return f"生成方案失败: {str(e)}"


def modify_feature_thinking(user_input: str, current_thinking: str) -> str:
    """修改特征工程方案"""
    return f"""根据 "{user_input}" 调整特征工程方案：

方案已更新。

---

**请回复 "确认" 执行**"""


async def generate_model_thinking(df: pd.DataFrame, target: str) -> str:
    """使用LLM动态生成建模方案"""
    import json
    
    # 判断任务类型
    if target in df.columns:
        target_unique = df[target].nunique()
        target_dtype = str(df[target].dtype)
        is_classification = target_dtype == 'object' or target_unique < 10
    else:
        is_classification = False
    
    prompt = f"""作为ML工程师，生成建模方案：

数据: {len(df)}样本 × {len(df.columns)}特征
目标: {target}, 类型: {'分类' if is_classification else '回归'}

参考ML Engineering：
1. 模型选择: 树模型(鲁棒), 梯度提升(性能), 线性模型(基线)
2. 验证: 分层K折(分类), 时间划分(时序)
3. 评估: 分类(Accuracy/F1), 回归(RMSE/MAE/R²)
4. 防过拟合: 正则化、早停、交叉验证
5. 流程: 基线→复杂→验证

请生成完整建模方案。"""

    try:
        llm = get_llm_client(temperature=0.3)
        response = await llm.ainvoke([
            {"role": "system", "content": "你是机器学习专家。"},
            {"role": "user", "content": prompt}
        ])
        
        return f"""## 🤖 建模方案\n\n{response.content.strip()}\n\n---\n\n**请回复 "确认" 执行，或提出修改意见**"""
    except Exception as e:
        return f"生成方案失败: {str(e)}"


def modify_model_thinking(user_input: str, current_thinking: str) -> str:
    """修改建模方案"""
    return f"""根据 "{user_input}" 调整建模方案：

方案已更新。

---

**请回复 "确认" 执行**"""


def execute_cleaning(df: pd.DataFrame, custom_thinking: str = None, target: str = None) -> dict:
    """执行数据清洗"""
    df = df.copy()
    output_lines = ["## 数据清洗执行结果\n"]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 缺失值处理
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            output_lines.append(f"- {col}: 中位数填充 {median_val:.2f}")
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("未知")
            output_lines.append(f"- {col}: 填充'未知'")
    
    return {
        "success": True,
        "message": "清洗完成",
        "data": df,
        "output": "\n".join(output_lines) if len(output_lines) > 1 else "无需处理"
    }


def execute_feature_engineering(df: pd.DataFrame, custom_thinking: str = None) -> dict:
    """执行特征工程"""
    df = df.copy()
    output_lines = ["## 特征工程执行结果\n"]
    
    # 基础特征变换
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:3]:
        if df[col].skew() > 1.5:
            new_col = f"{col}_log"
            df[new_col] = np.log1p(df[col])
            output_lines.append(f"- 创建 {new_col}")
    
    return {
        "success": True,
        "message": "特征工程完成",
        "data": df,
        "output": "\n".join(output_lines) if len(output_lines) > 1 else "无需处理"
    }


def execute_model_training_with_model(df: pd.DataFrame, target: str) -> tuple:
    """执行模型训练"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    df_clean = df.dropna()
    if target not in df_clean.columns:
        return {"success": False, "message": f"目标列 {target} 不存在"}, None
    
    X = df_clean.drop(columns=[target] + [c for c in ['Id', 'ID'] if c in df_clean.columns])
    y = df_clean[target]
    
    # 编码类别特征
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.factorize(X[col])[0]
    
    X = X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    output = f"""## 模型训练结果

- RMSE: {rmse:.4f}
- R²: {r2:.4f}
- 测试样本: {len(y_test)}

模型已保存，可下载或生成报告。"""
    
    return {
        "success": True,
        "message": "训练完成",
        "output": output,
        "metrics": {"rmse": rmse, "r2": r2}
    }, model


def generate_full_pipeline_code(session_id: str, df: pd.DataFrame, target: str, model) -> str:
    """生成完整流程代码"""
    code = f'''# AutoML 完整建模流程
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载数据
df = pd.read_csv("your_data.csv")

# 2. 数据清洗
# TODO: 根据实际清洗方案添加代码

# 3. 特征工程
# TODO: 根据实际特征工程方案添加代码

# 4. 训练模型
X = df.drop(columns=["{target}"])
y = df["{target}"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 评估
y_pred = model.predict(X_test)
print(f"RMSE: {{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}}")
print(f"R²: {{r2_score(y_test, y_pred):.4f}}")
'''
    return code


def generate_analysis_report(session_id: str) -> str:
    """生成分析报告"""
    return "# 建模分析报告\n\n报告内容..."


def download_model(session_id: str):
    """下载模型"""
    return None


# 路由定义
from fastapi import APIRouter
router = APIRouter()

@router.post("/sessions")
async def create_session():
    session_id = str(uuid.uuid4())
    session_manager.messages[session_id] = []
    session_manager.current_stage[session_id] = "initial"
    session_manager.current_thinking[session_id] = {}
    return {"id": session_id, "message": "会话创建成功"}


@router.post("/data/upload")
async def upload_data(session_id: str = Form(...), target_column: str = Form(...), file: UploadFile = File(...)):
    if session_id not in session_manager.messages:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    try:
        df = pd.read_csv(file.file)
        session_manager.data[session_id] = df
        session_manager.target_column[session_id] = target_column
        
        profile = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns),
            "missing": df.isnull().sum().to_dict(),
            "target_column": target_column
        }
        
        return {"success": True, "profile": profile, "message": "数据上传成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 新的数据分析流程 - 使用LLM生成思路和代码

async def generate_analysis_thinking(df: pd.DataFrame, target: str = None) -> str:
    """
    使用LLM生成数据分析思路，参考data-analysis skill
    """
    import json
    
    # 准备数据摘要
    column_info = []
    for col in df.columns[:20]:
        info = {"name": col, "dtype": str(df[col].dtype)}
        info["missing"] = int(df[col].isnull().sum())
        info["missing_pct"] = round(float(df[col].isnull().sum() / len(df) * 100), 2)
        
        if df[col].dtype in ['int64', 'float64']:
            info["type"] = "numeric"
            info["mean"] = round(float(df[col].mean()), 2)
            info["std"] = round(float(df[col].std()), 2)
            info["skew"] = round(float(df[col].skew()), 2)
        else:
            info["type"] = "categorical"
            info["unique"] = int(df[col].nunique())
        column_info.append(info)
    
    prompt = f"""作为数据分析师，请为以下数据生成详细的数据分析思路：

数据概况:
- 样本数: {len(df)}
- 特征数: {len(df.columns)}
- 目标列: {target or '未指定'}

特征信息:
{json.dumps(column_info, ensure_ascii=False, indent=2)}

参考data-analysis skill的最佳实践：
1. **描述性统计**: 均值、中位数、标准差、分位数
2. **数据质量评估**: 缺失值、异常值、重复值
3. **分布分析**: 正态性、偏度、峰度
4. **相关性分析**: 特征间相关性、与目标变量的关系
5. **可视化**: 直方图、箱线图、散点图、热力图

请生成具体的数据分析思路，包括：
1. 要分析哪些指标
2. 使用什么统计方法
3. 生成哪些可视化图表
4. 分析的顺序和逻辑

请用中文输出，格式清晰。"""

    try:
        llm = get_llm_client(temperature=0.3)
        response = await llm.ainvoke([
            {"role": "system", "content": "你是资深数据分析师，精通统计分析和数据可视化。"},
            {"role": "user", "content": prompt}
        ])
        
        return response.content.strip()
    except Exception as e:
        logger.error(f"生成分析思路失败: {e}")
        return f"生成分析思路失败: {str(e)}"


async def generate_analysis_code(df: pd.DataFrame, analysis_thinking: str, target: str = None) -> str:
    '''
    根据分析思路生成Python代码
    '''
    column_list = list(df.columns)
    
    prompt = f'''根据以下数据分析思路，生成完整的Python代码：

数据分析思路:
{analysis_thinking}

数据信息:
- 列名: {column_list}
- 目标列: {target or 'None'}
- 数据框变量名: df

要求:
1. 使用pandas进行数据处理
2. 使用matplotlib和seaborn进行可视化
3. 所有图表保存为base64字符串
4. 返回一个包含所有分析结果的字典
5. 代码必须完整可执行
6. 包含异常处理

代码模板结构:
[代码模板示例省略]

请只输出Python代码，不要输出其他内容。"""

    try:
        llm = get_llm_client(temperature=0.2)
        response = await llm.ainvoke([
            {"role": "system", "content": "你是Python数据分析师，精通pandas、matplotlib、seaborn。"},
            {"role": "user", "content": prompt}
        ])
        
        code = response.content.strip()
        # 提取代码块
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        return code.strip()
    except Exception as e:
        logger.error(f"生成分析代码失败: {e}")
        return None


def execute_analysis_code(code: str, df: pd.DataFrame, target: str = None) -> dict:
    """
    执行生成的分析代码
    """
    try:
        # 创建安全的执行环境
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        # 准备执行环境
        exec_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'base64': base64,
            'BytesIO': BytesIO,
            'df': df,
            'target': target
        }
        
        exec_locals = {}
        
        # 执行代码
        exec(code, exec_globals, exec_locals)
        
        # 调用分析函数
        if 'analyze_data' in exec_locals:
            results = exec_locals['analyze_data'](df, target)
        else:
            # 如果没有定义函数，尝试直接执行
            results = exec_locals.get('results', {})
        
        return {
            "success": True,
            "results": results,
            "code": code
        }
    except Exception as e:
        logger.error(f"执行分析代码失败: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "code": code
        }


async def perform_data_analysis(df: pd.DataFrame, target: str = None) -> dict:
    """
    完整的数据分析流程：思路 -> 代码 -> 执行
    """
    # 1. 生成分析思路
    thinking = await generate_analysis_thinking(df, target)
    
    # 2. 生成分析代码
    code = await generate_analysis_code(df, thinking, target)
    
    if not code:
        return {
            "success": False,
            "error": "生成分析代码失败",
            "thinking": thinking
        }
    
    # 3. 执行代码
    result = execute_analysis_code(code, df, target)
    
    return {
        "success": result["success"],
        "thinking": thinking,
        "code": code,
        "results": result.get("results", {}),
        "error": result.get("error", None)
    }


def format_analysis_results(analysis_result: dict, target: str = None) -> str:
    """
    格式化分析结果为Markdown报告
    """
    if not analysis_result["success"]:
        return f"""## ❌ 数据分析失败

错误信息: {analysis_result.get('error', '未知错误')}

分析思路:
{analysis_result.get('thinking', '无')}
"""
    
    results = analysis_result.get("results", {})
    thinking = analysis_result.get("thinking", "")
    
    report = f"""# 📊 数据分析报告

## 1. 分析思路

{thinking}

## 2. 分析结果

"""
    
    # 添加基础统计
    if "shape" in results:
        report += f"""### 数据概况
- 样本数: {results['shape'][0]}
- 特征数: {results['shape'][1]}

"""
    
    # 添加缺失值信息
    if "missing" in results:
        missing_items = {k: v for k, v in results["missing"].items() if v > 0}
        if missing_items:
            report += """### 缺失值情况
| 特征 | 缺失数 | 缺失率 |
|------|--------|--------|
"""
            for col, count in sorted(missing_items.items(), key=lambda x: x[1], reverse=True)[:10]:
                pct = count / results['shape'][0] * 100
                report += f"| {col} | {count} | {pct:.1f}% |\n"
            report += "\n"
    
    # 添加可视化
    report += "### 数据可视化\n\n"
    
    viz_keys = [k for k in results.keys() if k not in ['shape', 'missing', 'dtypes'] and isinstance(results[k], str)]
    
    for key in viz_keys:
        title = key.replace('_', ' ').title()
        report += f"#### {title}\n"
        report += f"![{title}](data:image/png;base64,{results[key]})\n\n"
    
    report += """## 3. 下一步建议

- 回复 "确认分析结果" 生成数据清洗方案
- 或提出您的具体需求

---

**请回复 "确认分析结果" 继续**
"""
    
    return report
