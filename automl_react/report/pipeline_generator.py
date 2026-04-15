"""
全流程脚本生成器模块

生成可独立运行的全流程建模脚本，支持 train 和 predict 两种模式。
"""

import json
import re
import textwrap
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..assets import get_asset_manager


class PipelineGenerator:
    """
    全流程脚本生成器

    收集各阶段代码，组装完整的建模脚本。
    生成的 pipeline.py 支持：
      - python pipeline.py --mode train --data raw.csv
      - python pipeline.py --mode predict --data new.csv --model output/model.pkl
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or "default"
        self.asset_manager = get_asset_manager(session_id=self.session_id)

    def collect_stage_codes(self) -> Dict[str, str]:
        """收集各阶段代码"""
        code_files = {
            "data_cleaning": "cleaning.py",
            "feature_engineering": "feature_engineering.py",
            "model_training": "model_training.py",
            "model_evaluation": "model_evaluation.py",
        }
        stages = {}
        for stage, filename in code_files.items():
            code = self.asset_manager.read_asset("code", filename)
            if code:
                stages[stage] = code
        return stages

    # ------------------------------------------------------------------
    # 代码提取辅助
    # ------------------------------------------------------------------

    def _extract_core_code(self, code: str) -> str:
        """从代码中提取核心逻辑（去除 markdown 围栏）"""
        code_blocks = re.findall(r'```python\n(.*?)\n```', code, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r'```\n(.*?)\n```', code, re.DOTALL)
        core = code_blocks[0] if code_blocks else code
        lines = [line for line in core.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip() + "\n"

    def _collect_imports(self, *code_blocks: str) -> str:
        """从多段代码中收集所有 import 语句并去重，正确处理多行 import。"""
        seen = set()
        imports = []

        for code in code_blocks:
            core = self._extract_core_code(code)
            lines = core.splitlines()
            i = 0
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    # 检测多行 import: from X import (
                    if '(' in stripped and ')' not in stripped:
                        # 收集到闭合 ) 为止
                        full_import = stripped
                        i += 1
                        while i < len(lines):
                            full_import += " " + lines[i].strip()
                            if ')' in lines[i]:
                                break
                            i += 1
                        # 拆成单个 import
                        m = re.match(r'from\s+([\w.]+)\s+import\s*\((.+)\)', full_import)
                        if m:
                            module = m.group(1)
                            names = [n.strip() for n in m.group(2).split(',') if n.strip()]
                            for name in names:
                                stmt = f"from {module} import {name}"
                                if stmt not in seen:
                                    seen.add(stmt)
                                    imports.append(stmt)
                    else:
                        if stripped not in seen:
                            seen.add(stripped)
                            imports.append(stripped)
                i += 1

        # 确保关键 import 存在
        essentials = [
            "import os", "import sys", "import json", "import pickle",
            "import warnings", "import argparse",
            "import numpy as np", "import pandas as pd",
        ]
        for imp in essentials:
            if imp not in seen:
                imports.insert(0, imp)
                seen.add(imp)
        return "\n".join(sorted(imports, key=lambda x: (x.startswith("from "), x)))

    def _strip_scaffolding(self, code: str) -> str:
        """
        从脚本代码中去除脚手架：import、sys.argv、if __name__、globals().update。
        并自动解构 def main() 包装（去掉 def main(): 行并 dedent 函数体）。
        """
        core = self._extract_core_code(code)
        lines = core.splitlines()
        result = []
        i = 0
        in_multiline_import = False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 跳过多行 import 的续行
            if in_multiline_import:
                if ')' in stripped:
                    in_multiline_import = False
                i += 1
                continue

            # 跳过 import 行
            if stripped.startswith("import ") or stripped.startswith("from "):
                if '(' in stripped and ')' not in stripped:
                    in_multiline_import = True
                i += 1
                continue

            # 跳过 sys.argv 行
            if "sys.argv" in stripped:
                i += 1
                continue

            # 跳过 globals().update({...}) — 可能跨多行（应已被生成阶段拦截）
            if "globals().update(" in stripped:
                print("[PipelineGenerator] WARNING: 检测到 globals().update()，应已在生成阶段被拦截")
                # 计算括号深度直到闭合
                depth = stripped.count('(') - stripped.count(')') + stripped.count('{') - stripped.count('}')
                i += 1
                while i < len(lines) and depth > 0:
                    s = lines[i].strip()
                    depth += s.count('(') - s.count(')') + s.count('{') - s.count('}')
                    i += 1
                continue

            # 跳过 if __name__ == "__main__": main() 块（应已被生成阶段拦截）
            if stripped.startswith("if __name__"):
                print("[PipelineGenerator] WARNING: 检测到 if __name__ 块，应已在生成阶段被拦截")
                i += 1
                # 跳过 if 块内的缩进体
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and not next_line.startswith("    ") and not next_line.startswith("\t"):
                        break
                    i += 1
                continue

            result.append(line)
            i += 1

        text = "\n".join(result)

        # 解构 def main(): 包装
        text = self._unwrap_main(text)

        # 去除首尾空行
        return text.strip()

    def _unwrap_main(self, code: str) -> str:
        """如果代码主体被 def main(): 包裹，去掉定义行并 dedent 函数体。（应已被生成阶段拦截）"""
        lines = code.splitlines()
        result = []
        i = 0
        found = False
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped == "def main():" or stripped == "def main() -> None:":
                if not found:
                    print("[PipelineGenerator] WARNING: 检测到 def main() 包装，应已在生成阶段被拦截")
                    found = True
                # 收集函数体并 dedent
                i += 1
                body_lines = []
                while i < len(lines):
                    next_line = lines[i]
                    # 函数体结束：遇到非空行且无缩进
                    if next_line.strip() and not next_line.startswith("    ") and not next_line.startswith("\t"):
                        break
                    body_lines.append(next_line)
                    i += 1
                dedented = textwrap.dedent("\n".join(body_lines))
                result.append(dedented)
            else:
                result.append(lines[i])
                i += 1
        return "\n".join(result)

    def _wrap_as_function(self, code: str, func_name: str) -> str:
        """将脚本代码包装为 def func_name(input_path, output_path): 函数。"""
        body = self._strip_scaffolding(code)
        if not body.strip():
            return f"def {func_name}(input_path, output_path):\n    pass\n"
        # 如果脚本内部也定义了同名函数，重命名为 _inner_<func_name>（应已被生成阶段拦截）
        inner_pattern = re.compile(rf'^(def\s+){func_name}(\s*\()', re.MULTILINE)
        if inner_pattern.search(body):
            print(f"[PipelineGenerator] WARNING: 检测到同名函数 {func_name}，应已在生成阶段被拦截")
            inner_name = f"_inner_{func_name}"
            body = inner_pattern.sub(rf'\g<1>{inner_name}\2', body)
            # 同时替换调用处
            body = re.sub(
                rf'(?<![.\w]){func_name}\s*\(',
                f'{inner_name}(',
                body,
            )
        indented = "\n".join(
            "    " + line if line.strip() else ""
            for line in body.splitlines()
        )
        return f"def {func_name}(input_path, output_path):\n{indented}\n"

    def _parameterize_training_code(self, code: str) -> str:
        """从 model_training.py 提取核心训练逻辑，去掉路径赋值和脚手架。"""
        body = self._strip_scaffolding(code)
        # 去掉硬编码路径赋值行
        path_vars = {
            "train_split_path", "valid_split_path", "test_split_path",
            "model_path", "training_summary_path",
        }
        lines = []
        for line in body.splitlines():
            stripped = line.strip()
            skip = False
            for var in path_vars:
                if stripped.startswith(f"{var} =") or stripped.startswith(f'{var}='):
                    skip = True
                    break
            if skip:
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _load_split_config(self) -> Dict[str, Any]:
        """从 session 资产读取切分配置。"""
        raw = self.asset_manager.read_asset("analysis", "dataset_split_result.json")
        if raw:
            try:
                result = json.loads(raw)
                return {
                    "train_ratio": result.get("ratios", {}).get("train_ratio", 0.6),
                    "valid_ratio": result.get("ratios", {}).get("valid_ratio", 0.2),
                    "test_ratio": result.get("ratios", {}).get("test_ratio", 0.2),
                }
            except json.JSONDecodeError:
                pass
        return {"train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2}

    # ------------------------------------------------------------------
    # 主生成方法
    # ------------------------------------------------------------------

    def generate_pipeline_script(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "regression",
    ) -> str:
        stage_codes = self.collect_stage_codes()
        split_config = self._load_split_config()

        # 收集 imports
        all_codes = list(stage_codes.values())
        imports_block = self._collect_imports(*all_codes)

        # 包装 cleaning / FE 为函数
        clean_func = "def clean_data(input_path, output_path):\n    pass\n"
        if "data_cleaning" in stage_codes:
            clean_func = self._wrap_as_function(stage_codes["data_cleaning"], "clean_data")

        fe_func = "def engineer_features(input_path, output_path):\n    pass\n"
        if "feature_engineering" in stage_codes:
            fe_func = self._wrap_as_function(stage_codes["feature_engineering"], "engineer_features")

        # 提取训练核心代码并缩进到 run_training 函数内
        training_body = ""
        if "model_training" in stage_codes:
            raw_training = self._parameterize_training_code(stage_codes["model_training"])
            training_body = _indent(raw_training, 4)

        train_ratio = split_config.get("train_ratio", 0.6)
        valid_ratio = split_config.get("valid_ratio", 0.2)
        test_ratio = split_config.get("test_ratio", 0.2)

        # 组装脚本（不使用 f-string 嵌入大段代码，改用拼接避免缩进/转义问题）
        parts = []

        # -- header --
        parts.append(f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML 全流程 Pipeline 脚本

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
会话ID: {self.session_id}

使用说明:
  训练模式:  python pipeline.py --mode train --data raw_data.csv
  推理模式:  python pipeline.py --mode predict --data new_data.csv --model output/model.pkl
"""

{imports_block}
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================
# 命令行参数
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data", required=True, help="输入数据路径 (CSV)")
    parser.add_argument("--model", default=None, help="模型路径 (predict 模式必需)")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--target", default="{target_column}", help="目标列名")
    return parser.parse_args()

''')

        # -- clean_data --
        parts.append("# ============================================")
        parts.append("# 阶段 1: 数据清洗")
        parts.append("# ============================================")
        parts.append(clean_func)
        parts.append("")

        # -- engineer_features --
        parts.append("# ============================================")
        parts.append("# 阶段 2: 特征工程")
        parts.append("# ============================================")
        parts.append(fe_func)
        parts.append("")

        # -- run_training --
        run_training_header = f'''# ============================================
# 训练模式
# ============================================
def run_training(args):
    from sklearn.model_selection import train_test_split

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    model_dir = output_dir / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    target_column = args.target

    # ── 数据切分 ──
    print("=" * 60)
    print("[1/5] 数据切分")
    print("=" * 60)
    df = pd.read_csv(args.data)
    print(f"原始数据形状: {{df.shape}}")

    train_df, temp_df = train_test_split(df, test_size={1 - train_ratio}, random_state=42)
    relative_valid = {valid_ratio} / ({valid_ratio} + {test_ratio})
    valid_df, test_df = train_test_split(temp_df, test_size=1 - relative_valid, random_state=42)

    train_raw_path = str(data_dir / "train_raw.csv")
    valid_raw_path = str(data_dir / "valid_raw.csv")
    test_raw_path = str(data_dir / "test_raw.csv")
    train_df.to_csv(train_raw_path, index=False)
    valid_df.to_csv(valid_raw_path, index=False)
    test_df.to_csv(test_raw_path, index=False)
    print(f"切分完成: train={{len(train_df)}}, valid={{len(valid_df)}}, test={{len(test_df)}}")

    # ── 数据清洗 ──
    print("\\n" + "=" * 60)
    print("[2/5] 数据清洗")
    print("=" * 60)
    cleaned_train = str(data_dir / "cleaned_train.csv")
    cleaned_valid = str(data_dir / "cleaned_valid.csv")
    cleaned_test = str(data_dir / "cleaned_test.csv")
    clean_data(train_raw_path, cleaned_train)
    print("  训练集清洗完成")
    clean_data(valid_raw_path, cleaned_valid)
    print("  验证集清洗完成")
    clean_data(test_raw_path, cleaned_test)
    print("  测试集清洗完成")

    # ── 特征工程 ──
    print("\\n" + "=" * 60)
    print("[3/5] 特征工程")
    print("=" * 60)
    features_train = str(data_dir / "features_train.csv")
    features_valid = str(data_dir / "features_valid.csv")
    features_test = str(data_dir / "features_test.csv")
    engineer_features(cleaned_train, features_train)
    print("  训练集特征工程完成")
    engineer_features(cleaned_valid, features_valid)
    print("  验证集特征工程完成")
    engineer_features(cleaned_test, features_test)
    print("  测试集特征工程完成")

    # ── 模型训练 ──
    print("\\n" + "=" * 60)
    print("[4/5] 模型训练")
    print("=" * 60)
    train_split_path = features_train
    valid_split_path = features_valid
    test_split_path = features_test
    model_path = str(model_dir / "trained_model.pkl")
    training_summary_path = str(model_dir / "training_summary.json")

'''
        parts.append(run_training_header.rstrip())

        # 内联训练代码（已经缩进 4 格）
        if training_body.strip():
            parts.append(training_body)

        # 训练尾部 + 简易评估
        run_training_footer = '''
    print("  模型训练完成")
    print(f"  模型保存至: {model_path}")

    # ── 简易评估 (测试集) ──
    print("\\n" + "=" * 60)
    print("[5/5] 模型评估 (测试集)")
    print("=" * 60)
    try:
        eval_df = pd.read_csv(features_test)
        if target_column in eval_df.columns:
            y_true = eval_df[target_column]
            eval_X = eval_df.drop(columns=[target_column, "Id"], errors="ignore")
            artifact = pickle.load(open(model_path, "rb"))
            if isinstance(artifact, dict) and "model" in artifact:
                mdl = artifact["model"]
                prep = artifact.get("preprocessor")
                feat_names = artifact.get("selected_feature_names", [])
                tt = artifact.get("target_transform")
                if feat_names:
                    eval_X = eval_X.reindex(columns=feat_names, fill_value=0)
                eval_input = prep.transform(eval_X) if prep is not None else eval_X
                preds = mdl.predict(eval_input)
                if isinstance(tt, str) and "log1p" in tt.lower():
                    preds = np.expm1(preds)
                elif isinstance(tt, dict) and any("log1p" in str(v).lower() for v in tt.values()):
                    preds = np.expm1(preds)
            else:
                preds = artifact.predict(eval_X)

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            print(f"  测试集 RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        else:
            print(f"  测试集不包含目标列 {target_column}，跳过评估")
    except Exception as e:
        print(f"  评估失败: {e}")

    print("\\n" + "=" * 60)
    print("训练流程完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
'''
        parts.append(run_training_footer)

        # -- run_predict --
        run_predict_code = '''
# ============================================
# 推理模式
# ============================================
def run_predict(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    if not args.model:
        print("错误: 推理模式需要指定 --model 参数")
        sys.exit(1)

    model_path = args.model
    target_column = args.target

    print("=" * 60)
    print("[1/4] 加载模型")
    print("=" * 60)
    try:
        import joblib
        model_artifact = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model_artifact = pickle.load(f)
    print(f"模型加载成功: {model_path}")

    # ── 数据清洗 ──
    print("\\n" + "=" * 60)
    print("[2/4] 数据清洗")
    print("=" * 60)
    cleaned_path = str(tmp_dir / "cleaned.csv")
    clean_data(args.data, cleaned_path)

    # ── 特征工程 ──
    print("\\n" + "=" * 60)
    print("[3/4] 特征工程")
    print("=" * 60)
    features_path = str(tmp_dir / "features.csv")
    engineer_features(cleaned_path, features_path)

    # ── 预测 ──
    print("\\n" + "=" * 60)
    print("[4/4] 模型预测")
    print("=" * 60)
    df = pd.read_csv(features_path)

    drop_cols = [c for c in [target_column, "Id"] if c in df.columns]
    X = df.drop(columns=drop_cols) if drop_cols else df.copy()

    if isinstance(model_artifact, dict) and "model" in model_artifact:
        model = model_artifact["model"]
        preprocessor = model_artifact.get("preprocessor")
        feature_names = model_artifact.get("selected_feature_names", [])
        target_transform = model_artifact.get("target_transform")
    else:
        model = model_artifact
        preprocessor = None
        feature_names = []
        target_transform = None

    if feature_names:
        X = X.reindex(columns=feature_names, fill_value=0)
    X_input = preprocessor.transform(X) if preprocessor is not None else X
    predictions = model.predict(X_input)

    if isinstance(target_transform, dict):
        if any("log1p" in str(v).lower() for v in target_transform.values()):
            predictions = np.expm1(predictions)
    elif isinstance(target_transform, str) and "log1p" in target_transform.lower():
        predictions = np.expm1(predictions)

    result_df = pd.DataFrame({"prediction": predictions})
    if "Id" in df.columns:
        result_df.insert(0, "Id", df["Id"].values)

    output_path = str(output_dir / "predictions.csv")
    result_df.to_csv(output_path, index=False)
    print(f"预测完成: {len(predictions)} 条记录")
    print(f"结果保存至: {output_path}")

    if target_column in df.columns:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_true = df[target_column]
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        print(f"\\n评估指标: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
'''
        parts.append(run_predict_code)

        # -- entry point --
        parts.append('''
# ============================================
# 入口
# ============================================
if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "predict":
        run_predict(args)
''')

        script = "\n".join(parts)
        # 压缩过多连续空行
        script = re.sub(r'\n{4,}', '\n\n\n', script)

        # 保存
        self.asset_manager.save_code(
            code=script,
            filename="pipeline.py",
            metadata={
                "stage": "pipeline",
                "data_path": data_path,
                "target_column": target_column,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return script

    def get_pipeline_script(self) -> Optional[str]:
        """获取已生成的全流程脚本"""
        return self.asset_manager.read_asset("code", "pipeline.py")

    def save_pipeline_script(self, script: str) -> Dict[str, Any]:
        """保存全流程脚本"""
        result = self.asset_manager.save_code(
            code=script,
            filename="pipeline.py",
            metadata={
                "stage": "pipeline",
                "timestamp": datetime.now().isoformat(),
            }
        )
        return {"success": True, "path": result.path, "size": result.size}


# ------------------------------------------------------------------
# 模块级辅助
# ------------------------------------------------------------------

def _indent(code: str, spaces: int) -> str:
    """给代码块的每一非空行增加缩进。"""
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line.strip() else ""
        for line in code.splitlines()
    )
