"""
流程导出器 - 将完整的 AutoML 流程导出为可执行的 Python 文件
"""
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd


class PipelineExporter:
    """流程导出器 - 从 PipelineState 中提取工作流并生成 Python 代码"""
    
    def __init__(self, state=None, pipeline=None):
        """
        初始化导出器
        
        Args:
            state: PipelineState 实例
            pipeline: Pipeline 实例
        """
        self.state = state
        self.pipeline = pipeline
        
    def extract_workflow(self) -> Dict[str, Any]:
        """从状态中提取完整工作流"""
        workflow = {
            "goal": self.state.goal if self.state else "",
            "target_column": self.state.target_column if self.state else "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_info": {
                "shape": list(self.state.data.shape) if self.state and self.state.data is not None else None,
                "columns": list(self.state.data.columns) if self.state and self.state.data is not None else [],
                "features": self.state.features if self.state else []
            },
            "steps": self._extract_steps(),
            "models": self._extract_models(),
            "metrics": self.state.results if self.state else {},
            "tool_calls": self.state.tool_calls if hasattr(self.state, 'tool_calls') else []
        }
        
        return workflow
    
    def _extract_steps(self) -> List[Dict[str, Any]]:
        """从日志中提取执行步骤"""
        steps = []
        
        if not self.state or not hasattr(self.state, 'logs'):
            return steps
        
        for log in self.state.logs:
            step_info = self._parse_log_to_step(log)
            if step_info:
                steps.append(step_info)
        
        return steps
    
    def _parse_log_to_step(self, log: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将日志解析为步骤信息"""
        message = log.get('message', '')
        
        # 识别关键步骤
        if '进入步骤' in message:
            return {
                "type": "step_enter",
                "step": message.split('：')[-1] if '：' in message else message
            }
        elif '数据设置完成' in message:
            return {
                "type": "data_loaded",
                "message": message
            }
        elif '新增特征' in message:
            return {
                "type": "feature_added",
                "feature_name": message.split('：')[-1] if '：' in message else ""
            }
        elif '模型已设置' in message:
            return {
                "type": "model_trained",
                "model_name": message.split('：')[-1] if '：' in message else ""
            }
        
        return None
    
    def _extract_models(self) -> Dict[str, Any]:
        """提取模型信息"""
        models_info = {}
        
        if not self.state or not hasattr(self.state, 'models'):
            return models_info
        
        for name, model in self.state.models.items():
            models_info[name] = {
                "type": type(model).__name__,
                "params": model.get_params() if hasattr(model, 'get_params') else {}
            }
        
        return models_info
    
    def generate_code(self, workflow: Dict[str, Any]) -> Dict[str, str]:
        """
        生成 Python 代码
        
        Args:
            workflow: 工作流信息
            
        Returns:
            代码字典 {文件名：代码内容}
        """
        codes = {}
        
        # 生成主流程文件
        codes["pipeline_main.py"] = self._generate_main_code(workflow)
        
        # 生成数据处理文件
        codes["data_preprocessing.py"] = self._generate_data_code(workflow)
        
        # 生成特征工程文件
        codes["feature_engineering.py"] = self._generate_feature_code(workflow)
        
        # 生成模型训练文件
        codes["model_training.py"] = self._generate_model_code(workflow)
        
        # 生成模型评估文件
        codes["model_evaluation.py"] = self._generate_eval_code(workflow)
        
        # 生成依赖文件
        codes["requirements.txt"] = self._generate_requirements()
        
        # 生成 README
        codes["README.md"] = self._generate_readme(workflow)
        
        return codes
    
    def _generate_main_code(self, workflow: Dict[str, Any]) -> str:
        """生成主流程代码"""
        code = f'''"""
AutoML Pipeline - 主流程文件
生成时间：{workflow['timestamp']}
建模目标：{workflow['goal']}
"""

import pandas as pd
import numpy as np
from data_preprocessing import load_and_preprocess_data
from feature_engineering import engineer_features
from model_training import train_models
from model_evaluation import evaluate_models


def main():
    """主函数"""
    print("=" * 60)
    print("AutoML Pipeline - 开始执行")
    print("=" * 60)
    
    # Step 1: 数据加载和预处理
    print("\\n[1/4] 加载和预处理数据...")
    df = load_and_preprocess_data("data.csv")
    print(f"  数据形状：{{df.shape}}")
    
    # Step 2: 特征工程
    print("\\n[2/4] 特征工程...")
    df = engineer_features(df)
    print(f"  特征数量：{{len(df.columns) - 1}}")
    
    # Step 3: 模型训练
    print("\\n[3/4] 训练模型...")
    models = train_models(df, target_column="{workflow['target_column']}")
    print(f"  训练完成：{{list(models.keys())}}")
    
    # Step 4: 模型评估
    print("\\n[4/4] 评估模型...")
    results = evaluate_models(models, df, target_column="{workflow['target_column']}")
    
    # 输出结果
    print("\\n" + "=" * 60)
    print("建模完成!")
    print("=" * 60)
    
    if "best_model_name" in results:
        print(f"最佳模型：{{results['best_model_name']}}")
    
    return models, results


if __name__ == "__main__":
    models, results = main()
'''
        return code
    
    def _generate_data_code(self, workflow: Dict[str, Any]) -> str:
        """生成数据处理代码"""
        code = f'''"""
AutoML Pipeline - 数据预处理模块
生成时间：{workflow['timestamp']}
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    加载数据文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        pd.DataFrame: 加载的数据
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的文件格式：{{filepath}}")
    
    print(f"  加载数据：{{df.shape}}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗数据
    
    Args:
        df: 原始数据
        
    Returns:
        pd.DataFrame: 清洗后的数据
    """
    df_clean = df.copy()
    
    # 处理缺失值
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # 删除重复值
    df_clean = df_clean.drop_duplicates()
    
    print(f"  清洗后数据：{{df_clean.shape}}")
    return df_clean


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    加载并预处理数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        pd.DataFrame: 预处理后的数据
    """
    # 加载数据
    df = load_data(filepath)
    
    # 清洗数据
    df = clean_data(df)
    
    return df
'''
        return code
    
    def _generate_feature_code(self, workflow: Dict[str, Any]) -> str:
        """生成特征工程代码"""
        features = workflow.get('data_info', {}).get('features', [])
        
        code = f'''"""
AutoML Pipeline - 特征工程模块
生成时间：{workflow['timestamp']}
"""

import pandas as pd
import numpy as np
from typing import List


def encode_categorical_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    编码分类特征
    
    Args:
        df: 输入数据
        exclude_cols: 需要排除的列
        
    Returns:
        pd.DataFrame: 编码后的数据
    """
    df_encoded = df.copy()
    
    if exclude_cols is None:
        exclude_cols = []
    
    # 识别分类列
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    # One-Hot 编码
    if categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=False)
        print(f"  编码分类特征：{{len(categorical_cols)}} 列")
    
    return df_encoded


def create_interaction_features(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
    """
    创建交互特征
    
    Args:
        df: 输入数据
        numeric_cols: 数值列列表
        
    Returns:
        pd.DataFrame: 添加交互特征后的数据
    """
    df_features = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建前几个数值列的交互特征
    count = 0
    for i, col1 in enumerate(numeric_cols[:3]):
        for col2 in numeric_cols[i+1:4]:
            feature_name = f"{{col1}}_mul_{{col2}}"
            df_features[feature_name] = df_features[col1] * df_features[col2]
            count += 1
    
    if count > 0:
        print(f"  创建交互特征：{{count}} 个")
    
    return df_features


def engineer_features(df: pd.DataFrame, target_column: str = "{workflow.get('target_column', 'target')}") -> pd.DataFrame:
    """
    执行特征工程
    
    Args:
        df: 输入数据
        target_column: 目标列
        
    Returns:
        pd.DataFrame: 特征工程后的数据
    """
    # 1. 编码分类特征
    df = encode_categorical_features(df, exclude_cols=[target_column])
    
    # 2. 创建交互特征
    df = create_interaction_features(df)
    
    print(f"  特征工程完成：{{df.shape}}")
    return df
'''
        return code
    
    def _generate_model_code(self, workflow: Dict[str, Any]) -> str:
        """生成模型训练代码"""
        models_info = workflow.get('models', {})
        model_names = list(models_info.keys()) if models_info else ['rf', 'xgb', 'lgbm']
        task_type = workflow.get('metrics', {}).get('task_type', 'classification')
        
        code = f'''"""
AutoML Pipeline - 模型训练模块
生成时间：{workflow['timestamp']}
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split


def train_model_rf(X_train, y_train):
    """训练随机森林模型"""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    if "{task_type}" == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    return model


def train_model_xgb(X_train, y_train):
    """训练 XGBoost 模型"""
    try:
        import xgboost as xgb
        
        if "{task_type}" == "classification":
            model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
        else:
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("  警告：xgboost 未安装，跳过")
        return None


def train_model_lgbm(X_train, y_train):
    """训练 LightGBM 模型"""
    try:
        import lightgbm as lgb
        
        if "{task_type}" == "classification":
            model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        else:
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("  警告：lightgbm 未安装，跳过")
        return None


def train_models(df: pd.DataFrame, target_column: str = "{workflow.get('target_column', 'target')}") -> Dict[str, Any]:
    """
    训练多个模型
    
    Args:
        df: 数据
        target_column: 目标列
        
    Returns:
        Dict: 训练好的模型字典
    """
    # 准备数据
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  训练集：{{X_train.shape}}, 测试集：{{X_test.shape}}")
    
    # 训练模型
    models = {{}}
    
    # 训练随机森林
    print("  - 训练随机森林...")
    models['rf'] = train_model_rf(X_train, y_train)
    
    # 训练 XGBoost
    print("  - 训练 XGBoost...")
    models['xgb'] = train_model_xgb(X_train, y_train)
    
    # 训练 LightGBM
    print("  - 训练 LightGBM...")
    models['lgbm'] = train_model_lgbm(X_train, y_train)
    
    # 移除 None 值
    models = {{k: v for k, v in models.items() if v is not None}}
    
    return models
'''
        return code
    
    def _generate_eval_code(self, workflow: Dict[str, Any]) -> str:
        """生成模型评估代码"""
        task_type = workflow.get('metrics', {}).get('task_type', 'classification')
        
        code = f'''"""
AutoML Pipeline - 模型评估模块
生成时间：{workflow['timestamp']}
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def evaluate_classification_model(y_true, y_pred, model_name: str) -> Dict[str, float]:
    """评估分类模型"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {{
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }}


def evaluate_regression_model(y_true, y_pred, model_name: str) -> Dict[str, float]:
    """评估回归模型"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    return {{
        "model": model_name,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }}


def evaluate_models(models: Dict, df: pd.DataFrame, target_column: str = "{workflow.get('target_column', 'target')}") -> Dict[str, Any]:
    """
    评估所有模型
    
    Args:
        models: 模型字典
        df: 数据
        target_column: 目标列
        
    Returns:
        Dict: 评估结果
    """
    from sklearn.model_selection import train_test_split
    
    # 准备数据
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 评估每个模型
    results = []
    best_score = -float('inf')
    best_model_name = None
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        if "{task_type}" == "classification":
            metrics = evaluate_classification_model(y_test, y_pred, name)
            score = metrics['accuracy']
        else:
            metrics = evaluate_regression_model(y_test, y_pred, name)
            score = metrics['r2']
        
        results.append(metrics)
        
        if score > best_score:
            best_score = score
            best_model_name = name
        
        print(f"  - {{name}}: {{score:.4f}}")
    
    return {{
        "all_results": results,
        "best_model_name": best_model_name,
        "best_score": best_score
    }}
'''
        return code
    
    def _generate_requirements(self) -> str:
        """生成依赖文件"""
        requirements = """# AutoML Pipeline Requirements
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
"""
        return requirements
    
    def _generate_readme(self, workflow: Dict[str, Any]) -> str:
        """生成 README 文件"""
        readme = f"""# AutoML Pipeline

生成时间：{workflow['timestamp']}
建模目标：{workflow['goal']}

## 文件说明

- `pipeline_main.py` - 主流程文件
- `data_preprocessing.py` - 数据预处理模块
- `feature_engineering.py` - 特征工程模块
- `model_training.py` - 模型训练模块
- `model_evaluation.py` - 模型评估模块
- `requirements.txt` - Python 依赖
- `data.csv` - 数据文件（需自行准备）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行流程

```bash
python pipeline_main.py
```

## 数据准备

将您的数据文件保存为 `data.csv`，确保包含目标列：`{workflow['target_column']}`

## 输出

运行完成后将显示：
- 数据形状
- 特征数量
- 训练的模型列表
- 每个模型的评估指标
- 最佳模型

## 自定义

您可以修改 `pipeline_main.py` 中的各个步骤来调整流程。
"""
        return readme
    
    def export_to_files(self, output_dir: str = "./exported_pipeline") -> str:
        """
        导出到文件
        
        Args:
            output_dir: 输出目录
            
        Returns:
            输出目录路径
        """
        # 提取工作流
        workflow = self.extract_workflow()
        
        # 生成代码
        codes = self.generate_code(workflow)
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{{output_dir}}_{{timestamp}}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存文件
        saved_files = []
        for filename, code in codes.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            saved_files.append(filename)
            print(f"  ✓ 保存文件：{filename}")
        
        print(f"\n✅ 流程已导出到：{output_dir}")
        print(f"   共 {len(saved_files)} 个文件：{', '.join(saved_files)}")
        
        return output_dir