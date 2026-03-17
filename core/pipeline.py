"""
AutoML 流程管理
"""
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field
from .state import PipelineState, StateStep


@dataclass
class PipelineConfig:
    """流程配置"""
    name: str = "AutoML Pipeline"
    version: str = "1.0.0"
    enable_logging: bool = True
    enable_checkpoint: bool = True
    checkpoint_dir: str = "./checkpoints"
    max_retries: int = 3
    timeout: int = 3600  # 秒
    

@dataclass
class PipelineStep:
    """流程步骤定义"""
    name: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    

class Pipeline:
    """AutoML 流程管理器"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        初始化流程管理器
        
        Args:
            config: 流程配置
        """
        self.config = config or PipelineConfig()
        self.steps: Dict[str, PipelineStep] = {}
        self.state: Optional[PipelineState] = None
        self.step_order: List[str] = []
        
    def set_state(self, state: PipelineState):
        """设置流程状态"""
        self.state = state
        self.state.log(f"流程初始化：{self.config.name}")
        
    def add_step(self, name: str, func: Callable, 
                 dependencies: Optional[List[str]] = None,
                 enabled: bool = True):
        """
        添加流程步骤
        
        Args:
            name: 步骤名称
            func: 步骤执行函数
            dependencies: 依赖的步骤名称列表
            enabled: 是否启用该步骤
        """
        self.steps[name] = PipelineStep(
            name=name,
            func=func,
            dependencies=dependencies or [],
            enabled=enabled
        )
        if name not in self.step_order:
            self.step_order.append(name)
            
    def remove_step(self, name: str):
        """移除步骤"""
        if name in self.steps:
            del self.steps[name]
            self.step_order.remove(name)
            
    def enable_step(self, name: str):
        """启用步骤"""
        if name in self.steps:
            self.steps[name].enabled = True
            
    def disable_step(self, name: str):
        """禁用步骤"""
        if name in self.steps:
            self.steps[name].enabled = False
            
    def _check_dependencies(self, step_name: str, completed_steps: set) -> bool:
        """检查步骤依赖是否已满足"""
        step = self.steps[step_name]
        for dep in step.dependencies:
            if dep not in completed_steps:
                return False
        return True
        
    def _get_next_step(self, completed_steps: set) -> Optional[str]:
        """获取下一个可执行的步骤"""
        for step_name in self.step_order:
            if step_name in completed_steps:
                continue
            step = self.steps[step_name]
            if not step.enabled:
                continue
            if self._check_dependencies(step_name, completed_steps):
                return step_name
        return None
        
    def execute(self, state: Optional[PipelineState] = None) -> PipelineState:
        """
        执行流程
        
        Args:
            state: 流程状态，如果不提供则使用已设置的状态
            
        Returns:
            PipelineState: 执行后的状态
        """
        if state is None:
            state = self.state
            
        if state is None:
            raise ValueError("必须提供流程状态")
            
        self.state = state
        completed_steps = set()
        
        try:
            while True:
                next_step = self._get_next_step(completed_steps)
                if next_step is None:
                    break
                    
                step = self.steps[next_step]
                self.state.log(f"开始执行步骤：{step.name}")
                
                try:
                    # 执行步骤
                    result = step.func(self.state)
                    completed_steps.add(next_step)
                    self.state.log(f"步骤完成：{step.name}")
                    
                except Exception as e:
                    self.state.log(f"步骤失败：{step.name} - {str(e)}", level="ERROR")
                    state.update_step(StateStep.FAILED)
                    raise
                    
        except Exception as e:
            self.state.log(f"流程执行失败：{str(e)}", level="ERROR")
            raise
            
        state.update_step(StateStep.COMPLETED)
        self.state.log("流程执行完成")
        return state
        
    def execute_step(self, step_name: str, state: Optional[PipelineState] = None) -> Any:
        """
        执行单个步骤
        
        Args:
            step_name: 步骤名称
            state: 流程状态
            
        Returns:
            步骤执行结果
        """
        if state is None:
            state = self.state
            
        if state is None:
            raise ValueError("必须提供流程状态")
            
        if step_name not in self.steps:
            raise ValueError(f"步骤不存在：{step_name}")
            
        step = self.steps[step_name]
        self.state.log(f"执行步骤：{step.name}")
        
        try:
            result = step.func(state)
            self.state.log(f"步骤完成：{step.name}")
            return result
        except Exception as e:
            self.state.log(f"步骤失败：{step.name} - {str(e)}", level="ERROR")
            raise
            
    def get_step_info(self, step_name: str) -> Dict[str, Any]:
        """获取步骤信息"""
        if step_name not in self.steps:
            raise ValueError(f"步骤不存在：{step_name}")
            
        step = self.steps[step_name]
        return {
            "name": step.name,
            "enabled": step.enabled,
            "dependencies": step.dependencies
        }
        
    def list_steps(self) -> List[Dict[str, Any]]:
        """列出所有步骤"""
        return [self.get_step_info(name) for name in self.step_order]