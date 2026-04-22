"""
沙盒执行器

提供统一的代码沙盒执行接口，支持 subprocess 模式（默认）和 Docker 模式（可选）。
借鉴 DeerFlow 的沙盒执行环境设计。
"""

from typing import Any, Dict, List, Optional

from .subprocess_executor import SubprocessCodeExecutor, SubprocessExecutionResult


class SandboxExecutor:
    """
    可配置的沙盒执行器。

    mode:
        - "subprocess" (默认): 使用 SubprocessCodeExecutor，带资源限制
        - "docker": 使用 Docker 容器执行（需要 docker 包，可选功能）
    """

    def __init__(
        self,
        mode: str = "subprocess",
        timeout: int = 300,
        python_path: Optional[str] = None,
        memory_limit_mb: int = 2048,
        cpu_time_limit: Optional[int] = None,
        **kwargs,
    ):
        self.mode = mode
        if mode == "subprocess":
            self._executor = SubprocessCodeExecutor(
                timeout=timeout,
                python_path=python_path,
                memory_limit_mb=memory_limit_mb,
                cpu_time_limit=cpu_time_limit,
            )
        elif mode == "docker":
            self._executor = _DockerExecutor(
                timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                **kwargs,
            )
        else:
            raise ValueError(f"不支持的沙盒模式: {mode}，可选: subprocess, docker")

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        required_output_names: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
    ) -> SubprocessExecutionResult:
        """执行代码并返回结果。"""
        return self._executor.execute(
            code=code,
            context=context,
            required_output_names=required_output_names,
            working_dir=working_dir,
        )


class _DockerExecutor:
    """
    Docker 容器沙盒执行器（可选功能）。

    需要安装 docker 包: pip install docker
    仅在显式配置 sandbox_mode="docker" 时使用。
    """

    def __init__(
        self,
        timeout: int = 300,
        memory_limit_mb: int = 2048,
        image: str = "python:3.11-slim",
        **kwargs,
    ):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.image = image

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        required_output_names: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
    ) -> SubprocessExecutionResult:
        """在 Docker 容器中执行代码。"""
        try:
            import docker
        except ImportError:
            return SubprocessExecutionResult(
                success=False,
                output="",
                error="Docker 模式需要安装 docker 包: pip install docker",
                variables={},
                return_code=-1,
                timed_out=False,
            )

        try:
            client = docker.from_env()
            container = client.containers.run(
                self.image,
                command=["python", "-c", code],
                mem_limit=f"{self.memory_limit_mb}m",
                network_disabled=True,
                remove=True,
                stdout=True,
                stderr=True,
                detach=False,
                timeout=self.timeout,
            )

            output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
            return SubprocessExecutionResult(
                success=True,
                output=output,
                error=None,
                variables={},
                return_code=0,
                timed_out=False,
            )
        except Exception as e:
            return SubprocessExecutionResult(
                success=False,
                output="",
                error=f"Docker 执行失败: {e}",
                variables={},
                return_code=-1,
                timed_out=False,
            )
