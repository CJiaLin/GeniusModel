"""
子进程代码执行器

在独立子进程中执行 LLM 生成的 Python 代码，提供：
- 超时强制终止
- stdout/stderr 捕获
- 崩溃隔离（子进程崩溃不影响主进程）
- 上下文变量通过 pickle 序列化传递
"""

import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SubprocessExecutionResult:
    """子进程执行结果"""
    success: bool
    output: str  # stdout
    error: Optional[str]  # stderr / 异常信息
    variables: Dict[str, Any]  # 反序列化的输出变量
    return_code: int
    timed_out: bool


class SubprocessCodeExecutor:
    """
    在子进程中执行 Python 代码。

    通过 pickle 临时文件传递上下文变量和收集输出变量，
    使用 subprocess.run 的 timeout 参数强制超时。
    支持可选的内存和 CPU 时间限制（通过 resource 模块）。
    """

    def __init__(
        self,
        timeout: int = 300,
        python_path: Optional[str] = None,
        memory_limit_mb: int = 2048,
        cpu_time_limit: Optional[int] = None,
    ):
        self.timeout = timeout
        self.python_path = python_path or sys.executable
        self.memory_limit_mb = memory_limit_mb
        self.cpu_time_limit = cpu_time_limit or timeout

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        required_output_names: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
    ) -> SubprocessExecutionResult:
        """
        在子进程中执行代码。

        Args:
            code: 要执行的 Python 代码
            context: 注入到执行环境的上下文变量（如 data_path 等）
            required_output_names: 需要从子进程回传的变量名列表
            working_dir: 工作目录（默认使用当前目录）
        """
        context = context or {}
        required_output_names = required_output_names or []

        # 使用 TemporaryDirectory 确保清理
        with tempfile.TemporaryDirectory(prefix="codeact_") as tmpdir:
            ctx_path = os.path.join(tmpdir, "context.pkl")
            out_path = os.path.join(tmpdir, "output.pkl")
            code_path = os.path.join(tmpdir, "script.py")

            # 序列化上下文（过滤不可序列化的值）
            serializable_ctx = self._filter_serializable(context)
            with open(ctx_path, "wb") as f:
                pickle.dump(serializable_ctx, f)

            # 生成 wrapper 脚本
            wrapper = self._build_wrapper_script(
                code, ctx_path, out_path, required_output_names
            )
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(wrapper)

            # 确定工作目录
            cwd = working_dir or os.getcwd()

            # 执行子进程
            try:
                result = subprocess.run(
                    [self.python_path, code_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=cwd,
                    env=self._build_env(),
                )

                stdout = result.stdout
                stderr = result.stderr
                return_code = result.returncode

            except subprocess.TimeoutExpired as e:
                return SubprocessExecutionResult(
                    success=False,
                    output=e.stdout or "" if hasattr(e, "stdout") and e.stdout else "",
                    error=f"执行超时: 超过 {self.timeout} 秒限制",
                    variables={},
                    return_code=-1,
                    timed_out=True,
                )

            # 收集输出变量
            output_vars = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, "rb") as f:
                        output_vars = pickle.load(f)
                except Exception:
                    pass

            success = return_code == 0
            error_text = stderr.strip() if stderr and stderr.strip() else None

            return SubprocessExecutionResult(
                success=success,
                output=stdout,
                error=error_text if not success else None,
                variables=output_vars,
                return_code=return_code,
                timed_out=False,
            )

    def _filter_serializable(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """过滤上下文，只保留可 pickle 序列化的值。"""
        safe = {}
        for k, v in context.items():
            if isinstance(v, (str, int, float, bool, list, dict, tuple, type(None))):
                safe[k] = v
            else:
                try:
                    pickle.dumps(v)
                    safe[k] = v
                except Exception:
                    safe[k] = str(v)
        return safe

    def _build_wrapper_script(
        self,
        user_code: str,
        ctx_path: str,
        out_path: str,
        required_output_names: List[str],
    ) -> str:
        """构建包含上下文加载和输出收集的 wrapper 脚本。"""
        # 收集通用变量名 + 用户指定的变量名
        all_output_names = list(set(required_output_names + [
            "df", "df_cleaned", "df_features", "model", "metrics",
            "X", "y", "X_train", "X_test", "y_train", "y_test",
            "selected_feature_names", "model_path", "training_summary_path",
            "result", "report",
        ]))

        return f'''# -*- coding: utf-8 -*-
import pickle
import sys
import os
import traceback

# ====== 资源限制 ======
try:
    import resource
    # 内存限制 (bytes)
    _mem_limit = {self.memory_limit_mb} * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_mem_limit, _mem_limit))
    except (ValueError, resource.error):
        pass  # 某些平台不支持 RLIMIT_AS
    # CPU 时间限制 (seconds)
    _cpu_limit = {self.cpu_time_limit}
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_cpu_limit, _cpu_limit + 10))
    except (ValueError, resource.error):
        pass
except ImportError:
    pass  # Windows 不支持 resource 模块

# ====== 加载上下文 ======
try:
    with open({ctx_path!r}, "rb") as _ctx_f:
        _ctx = pickle.load(_ctx_f)
    globals().update(_ctx)
    # 注入 sys.argv 以支持参数化脚本
    if 'input_data_path' in _ctx or 'output_data_path' in _ctx:
        sys.argv = [sys.argv[0] if sys.argv else 'script.py']
        if 'input_data_path' in _ctx:
            sys.argv.append(_ctx['input_data_path'])
        if 'output_data_path' in _ctx:
            sys.argv.append(_ctx['output_data_path'])
except Exception as _e:
    print(f"警告: 加载上下文失败: {{_e}}", file=sys.stderr)
    _ctx = {{}}

# ====== 用户代码 ======
try:
{_indent_code(user_code, 4)}
except Exception as _e:
    print(f"{{type(_e).__name__}}: {{str(_e)}}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

# ====== 收集输出变量 ======
_output = {{}}
for _name in {all_output_names!r}:
    if _name in globals():
        _val = globals()[_name]
        try:
            pickle.dumps(_val)
            _output[_name] = _val
        except Exception:
            _output[_name] = str(_val)

try:
    with open({out_path!r}, "wb") as _out_f:
        pickle.dump(_output, _out_f)
except Exception as _e:
    print(f"警告: 保存输出变量失败: {{_e}}", file=sys.stderr)
'''

    def _build_env(self) -> Dict[str, str]:
        """构建子进程环境变量，继承当前环境但过滤敏感变量。"""
        env = os.environ.copy()
        # 过滤敏感变量，防止 LLM 生成的代码泄露密钥
        sensitive_prefixes = ("OPENAI_", "ANTHROPIC_", "AWS_SECRET", "DASHSCOPE_")
        for key in list(env.keys()):
            if any(key.startswith(prefix) for prefix in sensitive_prefixes):
                del env[key]
        return env


def _indent_code(code: str, spaces: int) -> str:
    """为代码添加缩进。"""
    indent = " " * spaces
    lines = code.split("\n")
    return "\n".join(indent + line for line in lines)
