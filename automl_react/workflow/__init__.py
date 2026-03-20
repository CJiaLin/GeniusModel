"""Workflow state management module.

This module provides state management for the AutoML workflow,
including stage tracking, history recording, context data management,
and state persistence.
"""

from automl_react.workflow.workflow_state import (
    WorkflowStage,
    WorkflowState,
    StateTransitionError,
    VALID_TRANSITIONS,
)

__all__ = [
    "WorkflowStage",
    "WorkflowState",
    "StateTransitionError",
    "VALID_TRANSITIONS",
]
