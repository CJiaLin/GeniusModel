"""
模型训练 Agent 集群 - 负责模型选择、训练、调优、评估等任务
"""
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from core.state import PipelineState, StateStep
import pandas as pd
import numpy as np


class ModelTrainingAgent(BaseAgent):
    """模型训练 Agent - 协调模型训练相关工作"""
    
    def __init__(self, llm=None, name="ModelTrainingAgent", verbose=False):
        super().__init__(llm, name, verbose)
        self.sub_agents = {
            "selector": ModelSelectorAgent(llm, verbose),
            "trainer": ModelTrainerAgent(llm, verbose),
            "tuner": ModelTunerAgent(llm, verbose),
            "evaluator": ModelEvaluatorAgent(llm, verbose)
        }
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """
        执行模型训练任务
        
        Args:
            state: 当前状态
            task: 任务描述
            
        Returns:
            更新后的状态
        """
        self.log(f"开始执行模型训练任务")
        state.update_step(StateStep.MODEL_TRAINING)
        
        # 根据任务类型调用相应的子 Agent
        task_type = task.get("type", "train")
        
        if task_type == "select":
            state = self.sub_agents["selector"].execute(state, task)
        elif task_type == "train":
            state = self.sub_agents["trainer"].execute(state, task)
        elif task_type == "tune":
            state = self.sub_agents["tuner"].execute(state, task)
        elif task_type == "evaluate":
            state = self.sub_agents["evaluator"].execute(state, task)
        else:
            # 执行完整的模型训练流程
            state = self._execute_full_pipeline(state, task)
        
        self.log(f"模型训练任务完成")
        return state
    
    def _execute_full_pipeline(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """执行完整的模型训练流程"""
        # 1. 选择模型
        state = self.sub_agents["selector"].execute(state, {"action": "select"})
        
        # 2. 训练模型
        state = self.sub_agents["trainer"].execute(state, {"action": "train"})
        
        # 3. 调优（可选）
        if task.get("enable_tuning", True):
            state = self.sub_agents["tuner"].execute(state, {"action": "tune"})
        
        # 4. 评估
        state = self.sub_agents["evaluator"].execute(state, {"action": "evaluate"})
        
        return state


class ModelSelectorAgent(BaseAgent):
    """模型选择 Agent - 根据任务选择合适的模型"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """选择模型"""
        self.log("开始选择模型")
        
        # 确定任务类型
        task_type = self._determine_task_type(state)
        
        # 基于 LLM 或规则选择模型
        if self.llm and task.get("strategy", "auto") == "llm":
            selected_models = self._llm_select_models(state, task_type)
        else:
            selected_models = self._rule_based_select(state, task_type)
        
        # 保存选择结果
        state.set_result("selected_models", selected_models)
        state.set_result("task_type", task_type)
        
        self.log(f"模型选择完成：{[m['name'] for m in selected_models]}")
        return state
    
    def _determine_task_type(self, state: PipelineState) -> str:
        """确定任务类型"""
        if state.data is None or state.target_column not in state.data.columns:
            return "unknown"
        
        target_data = state.data[state.target_column]
        
        # 分类任务：目标列是类别型或数值型但唯一值较少
        if target_data.dtype == 'object' or len(target_data.unique()) < 10:
            return "classification"
        
        # 回归任务：目标列是数值型
        elif np.issubdtype(target_data.dtype, np.number):
            return "regression"
        
        return "unknown"
    
    def _llm_select_models(self, state: PipelineState, task_type: str) -> List[Dict[str, Any]]:
        """基于 LLM 选择模型"""
        prompt = f"""
根据以下信息，推荐 3-5 个合适的机器学习模型：

任务类型：{task_type}
数据形状：{state.data.shape if state.data is not None else '未知'}
特征数量：{len(state.features)}
建模目标：{state.goal}

可用模型：
- 随机森林 (rf)
- XGBoost (xgb)
- LightGBM (lgbm)
- 逻辑回归 (lr)
- 支持向量机 (svc)
- 线性回归 (linear)

请推荐模型并说明理由，以 JSON 格式返回：
[
    {{"name": "rf", "reason": "理由", "priority": "high"}},
    ...
]
"""
        
        try:
            response = self.invoke_llm(prompt)
            import json
            models = json.loads(response)
            return models if isinstance(models, list) else []
        except Exception as e:
            self.log(f"LLM 模型选择失败：{str(e)}", level="WARNING")
            return self._rule_based_select(state, task_type)
    
    def _rule_based_select(self, state: PipelineState, task_type: str) -> List[Dict[str, Any]]:
        """基于规则选择模型"""
        if task_type == "classification":
            return [
                {"name": "rf", "reason": "适用于分类任务，鲁棒性好", "priority": "high"},
                {"name": "xgb", "reason": "高性能梯度提升", "priority": "high"},
                {"name": "lgbm", "reason": "快速训练，适合大数据", "priority": "medium"},
                {"name": "lr", "reason": "基线模型，可解释性好", "priority": "low"}
            ]
        elif task_type == "regression":
            return [
                {"name": "rf", "reason": "适用于回归任务", "priority": "high"},
                {"name": "xgb", "reason": "高性能梯度提升回归", "priority": "high"},
                {"name": "lgbm", "reason": "快速回归训练", "priority": "medium"}
            ]
        else:
            return []


class ModelTrainerAgent(BaseAgent):
    """模型训练 Agent - 训练选定的模型"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """训练模型"""
        self.log("开始训练模型")
        
        from tools.model_tools import train_model
        from tools.data_tools import split_data
        
        if state.data is None:
            raise ValueError("数据未加载")
        
        # 获取训练参数
        test_size = task.get("test_size", 0.2)
        random_state = task.get("random_state", 42)
        
        # 分割数据
        X = state.data.drop(columns=[state.target_column])
        y = state.data[state.target_column]
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 保存测试集到状态
        state.set_result("X_test", X_test)
        state.set_result("y_test", y_test)
        state.set_result("y_test_true", y_test)
        
        # 获取选定的模型
        selected_models = state.results.get("selected_models", [])
        task_type = state.results.get("task_type", "classification")
        
        # 训练每个选定的模型
        trained_models = {}
        for model_info in selected_models:
            model_name = model_info.get("name", "unknown")
            
            try:
                # 训练模型
                model = train_model(
                    X_train, y_train,
                    model_type=model_name,
                    task_type=task_type,
                    params=model_info.get("params", {})
                )
                
                trained_models[model_name] = model
                state.set_model(model_name, model)
                
                self.log(f"模型训练完成：{model_name}")
                
            except Exception as e:
                self.log(f"模型训练失败：{model_name} - {str(e)}", level="ERROR")
        
        # 保存训练结果
        state.set_result("trained_models", trained_models)
        state.set_result("train_size", len(X_train))
        state.set_result("test_size", len(X_test))
        
        return state


class ModelTunerAgent(BaseAgent):
    """模型调优 Agent - 超参数优化"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """调优模型"""
        self.log("开始调优模型")
        
        from tools.model_tools import tune_hyperparameters
        
        # 获取要调优的模型
        model_name = task.get("model_name", "best")
        models_to_tune = task.get("models", [])
        
        if not models_to_tune and model_name == "best":
            # 选择最佳模型进行调优
            trained_models = state.results.get("trained_models", {})
            if trained_models:
                # 简单选择第一个模型
                model_name = list(trained_models.keys())[0]
                models_to_tune = [model_name]
        
        # 获取数据
        X_train = state.data.drop(columns=[state.target_column])
        y_train = state.data[state.target_column]
        task_type = state.results.get("task_type", "classification")
        
        # 为每个模型调优
        tuning_results = {}
        for name in models_to_tune:
            try:
                model = state.models.get(name)
                if model is None:
                    continue
                
                # 定义参数网格
                param_grid = self._get_param_grid(name)
                
                # 执行调优
                tuning_result = tune_hyperparameters(
                    model, X_train, y_train,
                    param_grid=param_grid,
                    cv=3
                )
                
                # 更新模型
                state.set_model(f"{name}_tuned", tuning_result["best_model"])
                tuning_results[name] = tuning_result
                
                self.log(f"模型调优完成：{name}, 最佳分数：{tuning_result['best_score']:.3f}")
                
            except Exception as e:
                self.log(f"模型调优失败：{name} - {str(e)}", level="WARNING")
        
        state.set_result("tuning_results", tuning_results)
        return state
    
    def _get_param_grid(self, model_name: str) -> Dict[str, List[Any]]:
        """获取参数网格"""
        if model_name == "rf":
            return {
                "n_estimators": [50, 100],
                "max_depth": [5, 10, None]
            }
        elif model_name == "xgb":
            return {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1]
            }
        elif model_name == "lgbm":
            return {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1]
            }
        else:
            return {}


class ModelEvaluatorAgent(BaseAgent):
    """模型评估 Agent - 评估模型性能"""
    
    def execute(self, state: PipelineState, task: Dict[str, Any]) -> PipelineState:
        """评估模型"""
        self.log("开始评估模型")
        
        from tools.eval_tools import evaluate_classification, evaluate_regression, compare_models
        
        # 获取测试集
        X_test = state.results.get("X_test")
        y_test = state.results.get("y_test")
        
        if X_test is None or y_test is None:
            self.log("测试集不存在，跳过评估", level="WARNING")
            return state
        
        # 获取所有训练好的模型
        models = state.models
        
        if not models:
            self.log("没有训练好的模型，跳过评估", level="WARNING")
            return state
        
        # 评估每个模型
        evaluation_results = {}
        task_type = state.results.get("task_type", "classification")
        
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                
                if task_type == "classification":
                    metrics = evaluate_classification(y_test, y_pred)
                else:
                    metrics = evaluate_regression(y_test, y_pred)
                
                evaluation_results[name] = {
                    "metrics": metrics,
                    "predictions": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
                }
                
                self.log(f"模型评估完成：{name}")
                
            except Exception as e:
                self.log(f"模型评估失败：{name} - {str(e)}", level="WARNING")
        
        # 比较模型
        if len(models) > 1:
            try:
                comparison = compare_models(models, X_test, y_test, task_type)
                state.set_result("model_comparison", comparison)
                
                # 找出最佳模型
                primary_metric = "accuracy" if task_type == "classification" else "r2"
                best_model = comparison.loc[comparison[primary_metric].idxmax()]
                state.set_result("best_model_name", best_model['model'])
                state.set_result("best_model", models[best_model['model']])
                
                self.log(f"最佳模型：{best_model['model']}, {primary_metric}: {best_model[primary_metric]:.3f}")
                
            except Exception as e:
                self.log(f"模型比较失败：{str(e)}", level="WARNING")
        
        # 保存评估结果
        state.set_result("evaluation_results", evaluation_results)
        state.set_result("evaluation_completed", True)
        
        return state
