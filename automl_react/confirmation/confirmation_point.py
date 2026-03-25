"""
User Confirmation Point Management Module

This module provides functionality for managing user confirmation points
during the AutoML workflow, including status tracking, queue management,
and timeout handling.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union


class ConfirmationStatus(Enum):
    """Status enumeration for confirmation points."""
    PENDING = "pending"       # Waiting for user response
    CONFIRMED = "confirmed"   # User confirmed the proposal
    MODIFIED = "modified"     # User provided modifications
    SKIPPED = "skipped"       # User skipped this confirmation point
    REJECTED = "rejected"     # User rejected the proposal


@dataclass
class SkillReference:
    """Reference to a skill used in the proposal."""
    skill_name: str
    skill_path: str
    reference_file: str
    relevant_content: Optional[str] = None


@dataclass
class UserResponse:
    """User response to a confirmation point."""
    status: ConfirmationStatus
    timestamp: datetime = field(default_factory=datetime.now)
    comment: Optional[str] = None
    modifications: Optional[str] = None
    modified_proposal: Optional[str] = None


@dataclass
class UserConfirmationPoint:
    """
    Represents a single user confirmation point in the workflow.
    
    Attributes:
        id: Unique identifier for the confirmation point
        stage: Workflow stage (e.g., "data_cleaning", "feature_engineering", "modeling")
        proposal_content: The proposal content in Markdown format
        skills_referenced: List of skills referenced in the proposal
        expected_outcome: Description of expected outcome
        user_response: User's response to this confirmation point
        created_at: Creation timestamp
        timeout_seconds: Timeout duration in seconds (None for no timeout)
        metadata: Additional metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage: str = ""
    proposal_content: str = ""
    skills_referenced: List[SkillReference] = field(default_factory=list)
    expected_outcome: str = ""
    user_response: Optional[UserResponse] = None
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_resolved(self) -> bool:
        """Check if the confirmation point has been resolved."""
        if self.user_response is None:
            return False
        return self.user_response.status in [
            ConfirmationStatus.CONFIRMED,
            ConfirmationStatus.MODIFIED,
            ConfirmationStatus.SKIPPED
        ]

    def is_rejected(self) -> bool:
        """Check if the confirmation point has been rejected."""
        if self.user_response is None:
            return False
        return self.user_response.status == ConfirmationStatus.REJECTED

    def is_expired(self) -> bool:
        """Check if the confirmation point has expired."""
        if self.timeout_seconds is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.timeout_seconds

    def set_user_response(
        self,
        status: ConfirmationStatus,
        comment: Optional[str] = None,
        modifications: Optional[str] = None,
        modified_proposal: Optional[str] = None
    ) -> None:
        """Set the user response for this confirmation point."""
        self.user_response = UserResponse(
            status=status,
            comment=comment,
            modifications=modifications,
            modified_proposal=modified_proposal
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the confirmation point to a dictionary."""
        return {
            "id": self.id,
            "stage": self.stage,
            "proposal_content": self.proposal_content,
            "skills_referenced": [
                {
                    "skill_name": s.skill_name,
                    "skill_path": s.skill_path,
                    "reference_file": s.reference_file,
                    "relevant_content": s.relevant_content
                }
                for s in self.skills_referenced
            ],
            "expected_outcome": self.expected_outcome,
            "user_response": {
                "status": self.user_response.status.value,
                "timestamp": self.user_response.timestamp.isoformat(),
                "comment": self.user_response.comment,
                "modifications": self.user_response.modifications,
                "modified_proposal": self.user_response.modified_proposal
            } if self.user_response else None,
            "created_at": self.created_at.isoformat(),
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserConfirmationPoint":
        """Create a UserConfirmationPoint from a dictionary."""
        skills = [
            SkillReference(
                skill_name=s["skill_name"],
                skill_path=s["skill_path"],
                reference_file=s["reference_file"],
                relevant_content=s.get("relevant_content")
            )
            for s in data.get("skills_referenced", [])
        ]

        user_response = None
        if data.get("user_response"):
            ur = data["user_response"]
            user_response = UserResponse(
                status=ConfirmationStatus(ur["status"]),
                timestamp=datetime.fromisoformat(ur["timestamp"]),
                comment=ur.get("comment"),
                modifications=ur.get("modifications"),
                modified_proposal=ur.get("modified_proposal")
            )

        return cls(
            id=data["id"],
            stage=data["stage"],
            proposal_content=data["proposal_content"],
            skills_referenced=skills,
            expected_outcome=data.get("expected_outcome", ""),
            user_response=user_response,
            created_at=datetime.fromisoformat(data["created_at"]),
            timeout_seconds=data.get("timeout_seconds"),
            metadata=data.get("metadata", {})
        )


class ConfirmationTimeoutError(Exception):
    """Exception raised when a confirmation point times out."""
    pass


class ConfirmationRejectedError(Exception):
    """Exception raised when a confirmation point is rejected by the user."""
    pass


class ConfirmationManager:
    """
    Manages a queue of user confirmation points.
    
    This class provides functionality to:
    - Add confirmation points to the queue
    - Wait for user responses with optional timeout
    - Handle user modifications and recovery
    - Manage confirmation point lifecycle
    """

    def __init__(self):
        self._queue: List[UserConfirmationPoint] = []
        self._current: Optional[UserConfirmationPoint] = None
        self._history: List[UserConfirmationPoint] = []
        self._waiting_event: Optional[asyncio.Event] = None
        self._response_callbacks: List[Callable[[UserConfirmationPoint], None]] = []
        self._timeout_callbacks: List[Callable[[UserConfirmationPoint], None]] = []

    @property
    def current(self) -> Optional[UserConfirmationPoint]:
        """Get the current active confirmation point."""
        return self._current

    @property
    def queue_size(self) -> int:
        """Get the number of pending confirmation points in the queue."""
        return len(self._queue)

    @property
    def history(self) -> List[UserConfirmationPoint]:
        """Get the history of all processed confirmation points."""
        return self._history.copy()

    def add_confirmation_point(
        self,
        stage: str,
        proposal_content: str,
        skills_referenced: Optional[List[SkillReference]] = None,
        expected_outcome: str = "",
        timeout_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserConfirmationPoint:
        """
        Add a new confirmation point to the queue.
        
        Args:
            stage: Workflow stage
            proposal_content: Proposal content in Markdown format
            skills_referenced: List of referenced skills
            expected_outcome: Description of expected outcome
            timeout_seconds: Optional timeout in seconds
            metadata: Additional metadata
            
        Returns:
            The created UserConfirmationPoint
        """
        point = UserConfirmationPoint(
            stage=stage,
            proposal_content=proposal_content,
            skills_referenced=skills_referenced or [],
            expected_outcome=expected_outcome,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {}
        )
        self._queue.append(point)
        return point

    def add_response_callback(
        self,
        callback: Callable[[UserConfirmationPoint], None]
    ) -> None:
        """Add a callback to be called when a response is received."""
        self._response_callbacks.append(callback)

    def add_timeout_callback(
        self,
        callback: Callable[[UserConfirmationPoint], None]
    ) -> None:
        """Add a callback to be called when a confirmation times out."""
        self._timeout_callbacks.append(callback)

    async def wait_for_confirmation(
        self,
        point_id: Optional[str] = None,
        auto_advance: bool = True
    ) -> UserConfirmationPoint:
        """
        Wait for user confirmation.
        
        Args:
            point_id: Optional specific confirmation point ID to wait for.
                     If None, processes the next point in the queue.
            auto_advance: Whether to automatically advance to next point
            
        Returns:
            The confirmed UserConfirmationPoint
            
        Raises:
            ConfirmationTimeoutError: If the confirmation times out
            ConfirmationRejectedError: If the user rejects the proposal
        """
        if point_id:
            point = self._find_point_by_id(point_id)
            if not point:
                raise ValueError(f"Confirmation point {point_id} not found")
        else:
            if not self._queue:
                raise ValueError("No confirmation points in queue")
            point = self._queue.pop(0)

        self._current = point
        self._waiting_event = asyncio.Event()

        try:
            if point.timeout_seconds:
                await asyncio.wait_for(
                    self._waiting_event.wait(),
                    timeout=point.timeout_seconds
                )
            else:
                await self._waiting_event.wait()

            if point.is_rejected():
                raise ConfirmationRejectedError(
                    f"Confirmation point {point.id} was rejected"
                )

            if auto_advance:
                self._advance_queue(point)

            return point

        except asyncio.TimeoutError:
            for callback in self._timeout_callbacks:
                callback(point)
            raise ConfirmationTimeoutError(
                f"Confirmation point {point.id} timed out after {point.timeout_seconds}s"
            )

    def submit_response(
        self,
        point_id: str,
        status: ConfirmationStatus,
        comment: Optional[str] = None,
        modifications: Optional[str] = None,
        modified_proposal: Optional[str] = None
    ) -> bool:
        """
        Submit a user response for a confirmation point.
        
        Args:
            point_id: The confirmation point ID
            status: The response status
            comment: Optional user comment
            modifications: Optional modification description
            modified_proposal: Optional modified proposal content
            
        Returns:
            True if the response was successfully submitted
        """
        point = self._find_point_by_id(point_id)
        if not point:
            return False

        point.set_user_response(status, comment, modifications, modified_proposal)

        if self._waiting_event and self._current and self._current.id == point_id:
            self._waiting_event.set()

        for callback in self._response_callbacks:
            callback(point)

        return True

    def modify_and_retry(
        self,
        point_id: str,
        modified_proposal: str,
        new_skills_referenced: Optional[List[SkillReference]] = None
    ) -> Optional[UserConfirmationPoint]:
        """
        Create a new confirmation point based on user modifications.
        
        This allows the workflow to continue with a modified proposal
        after user feedback.
        
        Args:
            point_id: The original confirmation point ID
            modified_proposal: The modified proposal content
            new_skills_referenced: Optional new skill references
            
        Returns:
            The new UserConfirmationPoint, or None if original not found
        """
        original = self._find_point_by_id(point_id)
        if not original:
            return None

        new_point = UserConfirmationPoint(
            stage=original.stage,
            proposal_content=modified_proposal,
            skills_referenced=new_skills_referenced or original.skills_referenced,
            expected_outcome=original.expected_outcome,
            timeout_seconds=original.timeout_seconds,
            metadata={
                **original.metadata,
                "parent_id": original.id,
                "is_retry": True
            }
        )

        self._queue.insert(0, new_point)
        return new_point

    def skip_current(self, comment: Optional[str] = None) -> bool:
        """
        Skip the current confirmation point.
        
        Args:
            comment: Optional comment for skipping
            
        Returns:
            True if successfully skipped
        """
        if not self._current:
            return False

        self._current.set_user_response(
            status=ConfirmationStatus.SKIPPED,
            comment=comment
        )

        if self._waiting_event:
            self._waiting_event.set()

        return True

    def cancel_current(self, reason: Optional[str] = None) -> bool:
        """
        Cancel the current confirmation point.
        
        Args:
            reason: Optional reason for cancellation
            
        Returns:
            True if successfully cancelled
        """
        if not self._current:
            return False

        self._current.set_user_response(
            status=ConfirmationStatus.REJECTED,
            comment=reason
        )

        if self._waiting_event:
            self._waiting_event.set()

        return True

    def get_pending_points(self) -> List[UserConfirmationPoint]:
        """Get all pending confirmation points."""
        return [p for p in self._queue if not p.is_resolved() and not p.is_rejected()]

    def get_points_by_stage(self, stage: str) -> List[UserConfirmationPoint]:
        """Get all confirmation points for a specific stage."""
        return [p for p in self._history + self._queue + ([self._current] if self._current else [])
                if p.stage == stage]

    def clear_queue(self) -> None:
        """Clear all pending confirmation points."""
        self._queue.clear()

    def clear_history(self) -> None:
        """Clear the confirmation point history."""
        self._history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the manager state to a dictionary."""
        return {
            "queue": [p.to_dict() for p in self._queue],
            "current": self._current.to_dict() if self._current else None,
            "history": [p.to_dict() for p in self._history]
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore the manager state from a dictionary."""
        self._queue = [UserConfirmationPoint.from_dict(p) for p in data.get("queue", [])]
        self._history = [UserConfirmationPoint.from_dict(p) for p in data.get("history", [])]
        current_data = data.get("current")
        self._current = UserConfirmationPoint.from_dict(current_data) if current_data else None

    def _find_point_by_id(self, point_id: str) -> Optional[UserConfirmationPoint]:
        """Find a confirmation point by ID."""
        if self._current and self._current.id == point_id:
            return self._current

        for point in self._queue:
            if point.id == point_id:
                return point

        for point in self._history:
            if point.id == point_id:
                return point

        return None

    def _advance_queue(self, completed_point: UserConfirmationPoint) -> None:
        """Advance the queue after a point is completed."""
        self._history.append(completed_point)
        if self._current == completed_point:
            self._current = None
        self._waiting_event = None


class ConfirmationPointBuilder:
    """
    Builder class for creating UserConfirmationPoint instances.
    
    Example:
        point = (ConfirmationPointBuilder()
                 .with_stage("data_cleaning")
                 .with_proposal("## Proposal\n...")
                 .with_skill("data-analysis", "/skills/data-analysis", "techniques.md")
                 .with_expected_outcome("Clean dataset")
                 .with_timeout(300)
                 .build())
    """

    def __init__(self):
        self._stage = ""
        self._proposal_content = ""
        self._skills_referenced: List[SkillReference] = []
        self._expected_outcome = ""
        self._timeout_seconds: Optional[float] = None
        self._metadata: Dict[str, Any] = {}

    def with_stage(self, stage: str) -> "ConfirmationPointBuilder":
        """Set the workflow stage."""
        self._stage = stage
        return self

    def with_proposal(self, content: str) -> "ConfirmationPointBuilder":
        """Set the proposal content (Markdown format)."""
        self._proposal_content = content
        return self

    def with_skill(
        self,
        skill_name: str,
        skill_path: str,
        reference_file: str,
        relevant_content: Optional[str] = None
    ) -> "ConfirmationPointBuilder":
        """Add a skill reference."""
        self._skills_referenced.append(SkillReference(
            skill_name=skill_name,
            skill_path=skill_path,
            reference_file=reference_file,
            relevant_content=relevant_content
        ))
        return self

    def with_expected_outcome(self, outcome: str) -> "ConfirmationPointBuilder":
        """Set the expected outcome description."""
        self._expected_outcome = outcome
        return self

    def with_timeout(self, seconds: float) -> "ConfirmationPointBuilder":
        """Set the timeout duration."""
        self._timeout_seconds = seconds
        return self

    def with_metadata(self, key: str, value: Any) -> "ConfirmationPointBuilder":
        """Add metadata."""
        self._metadata[key] = value
        return self

    def build(self) -> UserConfirmationPoint:
        """Build and return the UserConfirmationPoint."""
        return UserConfirmationPoint(
            stage=self._stage,
            proposal_content=self._proposal_content,
            skills_referenced=self._skills_referenced,
            expected_outcome=self._expected_outcome,
            timeout_seconds=self._timeout_seconds,
            metadata=self._metadata
        )
