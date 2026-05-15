"""Workflow state management module.

This module provides state management for the AutoML workflow,
including stage tracking, history recording, context data management,
and state persistence.
"""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class WorkflowStage(Enum):
    """Workflow stage enumeration.

    Represents the different stages of the AutoML workflow pipeline.
    """
    DATA_UPLOAD = "data_upload"
    PROBLEM_DEFINITION = "problem_definition"
    DATA_CONTRACT_CHECK = "data_contract_check"
    DATA_SPLITTING = "data_splitting"
    DATA_AGGREGATION = "data_aggregation"
    DATA_CLEANING = "data_cleaning"  # 包含数据质量分析
    DATA_EXPLORATION = "data_exploration"  # 探索性数据分析（基于清洗后数据）
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    COMPLETED = "completed"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


class WorkflowMode:
    """Workflow mode constants."""
    FULL = "full"
    SCHEMA_ONLY = "schema_only"
    FEATURE_ONLY = "feature_only"


# Define valid stage transitions (full mode — default)
VALID_TRANSITIONS: dict[WorkflowStage, list[WorkflowStage]] = {
    WorkflowStage.DATA_UPLOAD: [
        WorkflowStage.PROBLEM_DEFINITION,
        WorkflowStage.DATA_CLEANING,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.PROBLEM_DEFINITION: [
        WorkflowStage.DATA_AGGREGATION,
        WorkflowStage.DATA_CONTRACT_CHECK,
        WorkflowStage.DATA_CLEANING,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.DATA_AGGREGATION: [
        WorkflowStage.DATA_CONTRACT_CHECK,
        WorkflowStage.DATA_SPLITTING,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.DATA_CONTRACT_CHECK: [
        WorkflowStage.DATA_SPLITTING,
        WorkflowStage.DATA_CLEANING,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.DATA_SPLITTING: [
        WorkflowStage.DATA_CLEANING,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.DATA_CLEANING: [
        WorkflowStage.DATA_EXPLORATION,
        WorkflowStage.FEATURE_ENGINEERING,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.DATA_EXPLORATION: [
        WorkflowStage.FEATURE_ENGINEERING,
        WorkflowStage.MODEL_TRAINING,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.FEATURE_ENGINEERING: [
        WorkflowStage.DATA_EXPLORATION,
        WorkflowStage.MODEL_TRAINING,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.MODEL_TRAINING: [
        WorkflowStage.MODEL_EVALUATION,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.MODEL_EVALUATION: [
        WorkflowStage.COMPLETED,
        WorkflowStage.ERROR,
    ],
    WorkflowStage.COMPLETED: [],
    WorkflowStage.ERROR: [
        WorkflowStage.DATA_UPLOAD,
        WorkflowStage.PROBLEM_DEFINITION,
        WorkflowStage.DATA_AGGREGATION,
        WorkflowStage.DATA_CONTRACT_CHECK,
        WorkflowStage.DATA_SPLITTING,
        WorkflowStage.DATA_CLEANING,
        WorkflowStage.DATA_EXPLORATION,
        WorkflowStage.FEATURE_ENGINEERING,
        WorkflowStage.MODEL_TRAINING,
    ],
}

# Schema-only mode: DATA_UPLOAD → PROBLEM_DEFINITION → FEATURE_ENGINEERING → COMPLETED
TRANSITIONS_SCHEMA_ONLY: dict[WorkflowStage, list[WorkflowStage]] = {
    WorkflowStage.DATA_UPLOAD: [WorkflowStage.PROBLEM_DEFINITION, WorkflowStage.ERROR],
    WorkflowStage.PROBLEM_DEFINITION: [WorkflowStage.FEATURE_ENGINEERING, WorkflowStage.ERROR],
    WorkflowStage.FEATURE_ENGINEERING: [WorkflowStage.COMPLETED, WorkflowStage.ERROR],
    WorkflowStage.COMPLETED: [],
    WorkflowStage.ERROR: [WorkflowStage.DATA_UPLOAD, WorkflowStage.PROBLEM_DEFINITION, WorkflowStage.FEATURE_ENGINEERING],
}

# Feature-only mode: DATA_UPLOAD → DATA_EXPLORATION → FEATURE_ENGINEERING → COMPLETED
TRANSITIONS_FEATURE_ONLY: dict[WorkflowStage, list[WorkflowStage]] = {
    WorkflowStage.DATA_UPLOAD: [WorkflowStage.DATA_EXPLORATION, WorkflowStage.FEATURE_ENGINEERING, WorkflowStage.ERROR],
    WorkflowStage.DATA_EXPLORATION: [WorkflowStage.FEATURE_ENGINEERING, WorkflowStage.ERROR],
    WorkflowStage.FEATURE_ENGINEERING: [WorkflowStage.COMPLETED, WorkflowStage.ERROR],
    WorkflowStage.COMPLETED: [],
    WorkflowStage.ERROR: [WorkflowStage.DATA_UPLOAD, WorkflowStage.DATA_EXPLORATION, WorkflowStage.FEATURE_ENGINEERING],
}


def get_transitions_for_mode(mode: str) -> dict[WorkflowStage, list[WorkflowStage]]:
    """Get the valid transitions dict for a given workflow mode."""
    if mode == WorkflowMode.SCHEMA_ONLY:
        return TRANSITIONS_SCHEMA_ONLY
    elif mode == WorkflowMode.FEATURE_ONLY:
        return TRANSITIONS_FEATURE_ONLY
    return VALID_TRANSITIONS


def get_stages_for_mode(mode: str) -> list[str]:
    """Get the ordered stage list for a given workflow mode."""
    if mode == WorkflowMode.SCHEMA_ONLY:
        return ["data_upload", "problem_definition", "feature_engineering"]
    elif mode == WorkflowMode.FEATURE_ONLY:
        return ["data_upload", "data_exploration", "feature_engineering"]
    return [
        "data_upload", "problem_definition", "data_aggregation",
        "data_contract_check", "data_splitting", "data_cleaning",
        "data_exploration", "feature_engineering", "model_training",
    ]


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class WorkflowState:
    """Workflow state manager.

    Manages the current stage, history of transitions, and context data
    for a workflow session. Supports persistence to JSON files.

    Attributes:
        session_id: Unique identifier for the workflow session.
        current_stage: The current stage of the workflow.
        history: List of stage transitions with timestamps.
        context: Dictionary storing workflow context data.
        state_file_path: Path to the state persistence file.
    """

    def __init__(
        self,
        session_id: str,
        base_path: str = "assets",
        initial_stage: WorkflowStage = WorkflowStage.DATA_UPLOAD,
    ):
        """Initialize workflow state.

        Args:
            session_id: Unique identifier for the workflow session.
            base_path: Base directory for state persistence.
            initial_stage: Initial workflow stage.
        """
        self.session_id = session_id
        self.current_stage = initial_stage
        self.history: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self._state_file_path = Path(base_path) / session_id / "state" / "workflow_state.json"

        # Record initial stage
        self._record_transition(initial_stage, "Workflow initialized")

    @property
    def state_file_path(self) -> Path:
        """Get the state file path."""
        return self._state_file_path

    def _record_transition(self, stage: WorkflowStage, message: str = "") -> None:
        """Record a stage transition in history.

        Args:
            stage: The stage being transitioned to.
            message: Optional message describing the transition.
        """
        transition = {
            "stage": stage.value,
            "timestamp": datetime.now().isoformat(),
            "message": message,
        }
        self.history.append(transition)

    def _validate_transition(self, new_stage: WorkflowStage) -> bool:
        """Validate if a transition to the new stage is allowed.

        Args:
            new_stage: The target stage for transition.

        Returns:
            True if the transition is valid, False otherwise.
        """
        mode = self.context.get("workflow_mode", WorkflowMode.FULL)
        transitions = get_transitions_for_mode(mode)
        valid_next_stages = transitions.get(self.current_stage, [])
        return new_stage in valid_next_stages

    def transition_to(
        self,
        new_stage: WorkflowStage,
        message: str = "",
        force: bool = False,
    ) -> None:
        """Transition to a new workflow stage.

        Args:
            new_stage: The target stage to transition to.
            message: Optional message describing the transition.
            force: If True, bypass transition validation.

        Raises:
            StateTransitionError: If the transition is invalid and force is False.
        """
        if not force and not self._validate_transition(new_stage):
            mode = self.context.get("workflow_mode", WorkflowMode.FULL)
            transitions = get_transitions_for_mode(mode)
            raise StateTransitionError(
                f"Invalid transition from {self.current_stage.value} to {new_stage.value}. "
                f"Valid transitions: {[s.value for s in transitions.get(self.current_stage, [])]}"
            )

        old_stage = self.current_stage
        self.current_stage = new_stage
        self._record_transition(new_stage, message)

        # Auto-save on transition
        self.save()

    def can_transition_to(self, stage: WorkflowStage) -> bool:
        """Check if transition to a stage is allowed.

        Args:
            stage: The target stage to check.

        Returns:
            True if transition is valid, False otherwise.
        """
        return self._validate_transition(stage)

    def get_valid_transitions(self) -> list[WorkflowStage]:
        """Get list of valid next stages from current stage.

        Returns:
            List of valid WorkflowStage values.
        """
        mode = self.context.get("workflow_mode", WorkflowMode.FULL)
        transitions = get_transitions_for_mode(mode)
        return transitions.get(self.current_stage, []).copy()

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value.

        Args:
            key: Context key.
            value: Context value (must be JSON serializable).
        """
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: Context key.
            default: Default value if key not found.

        Returns:
            The context value or default.
        """
        return self.context.get(key, default)

    def update_context(self, data: dict[str, Any]) -> None:
        """Update multiple context values.

        Args:
            data: Dictionary of context key-value pairs.
        """
        self.context.update(data)

    def clear_context(self) -> None:
        """Clear all context data."""
        self.context.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary.

        Returns:
            Dictionary representation of the workflow state.
        """
        return {
            "session_id": self.session_id,
            "current_stage": self.current_stage.value,
            "history": self.history,
            "context": self.context,
            "last_updated": datetime.now().isoformat(),
        }

    def save(self) -> None:
        """Save state to JSON file.

        Creates the state directory if it doesn't exist.
        """
        # Ensure directory exists
        self._state_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save state
        with open(self._state_file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, session_id: str, base_path: str = "assets") -> Optional["WorkflowState"]:
        """Load workflow state from JSON file.

        Args:
            session_id: The session ID to load.
            base_path: Base directory for state persistence.

        Returns:
            WorkflowState instance if found, None otherwise.
        """
        state_file_path = Path(base_path) / session_id / "state" / "workflow_state.json"

        if not state_file_path.exists():
            return None

        with open(state_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.session_id = data["session_id"]
        instance.current_stage = WorkflowStage(data["current_stage"])
        instance.history = data.get("history", [])
        instance.context = data.get("context", {})
        instance._state_file_path = state_file_path

        return instance

    @classmethod
    def create_or_load(
        cls,
        session_id: str,
        base_path: str = "assets",
        initial_stage: WorkflowStage = WorkflowStage.DATA_UPLOAD,
    ) -> "WorkflowState":
        """Create new state or load existing one.

        Args:
            session_id: The session ID.
            base_path: Base directory for state persistence.
            initial_stage: Initial stage if creating new state.

        Returns:
            WorkflowState instance (loaded or newly created).
        """
        existing = cls.load(session_id, base_path)
        if existing:
            return existing
        return cls(session_id, base_path, initial_stage)

    def reset(self, initial_stage: WorkflowStage = WorkflowStage.DATA_UPLOAD) -> None:
        """Reset workflow state to initial stage.

        Args:
            initial_stage: Stage to reset to.
        """
        self.current_stage = initial_stage
        self.history.clear()
        self.context.clear()
        self._record_transition(initial_stage, "Workflow reset")
        self.save()

    def get_stage_history(self, stage: Optional[WorkflowStage] = None) -> list[dict[str, Any]]:
        """Get history entries, optionally filtered by stage.

        Args:
            stage: Optional stage to filter by.

        Returns:
            List of history entries.
        """
        if stage is None:
            return self.history.copy()
        return [h for h in self.history if h["stage"] == stage.value]

    def get_time_in_current_stage(self) -> Optional[float]:
        """Get time spent in current stage in seconds.

        Returns:
            Time in seconds since entering current stage, or None if no history.
        """
        if not self.history:
            return None

        # Find last entry for current stage
        for entry in reversed(self.history):
            if entry["stage"] == self.current_stage.value:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                return (datetime.now() - entry_time).total_seconds()

        return None

    def is_completed(self) -> bool:
        """Check if workflow is completed.

        Returns:
            True if current stage is COMPLETED.
        """
        return self.current_stage == WorkflowStage.COMPLETED

    def is_error(self) -> bool:
        """Check if workflow is in error state.

        Returns:
            True if current stage is ERROR.
        """
        return self.current_stage == WorkflowStage.ERROR

    def __repr__(self) -> str:
        return f"WorkflowState(session_id={self.session_id}, stage={self.current_stage.value})"
