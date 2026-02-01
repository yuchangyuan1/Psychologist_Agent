"""
Audit Logger for API calls and system events.

This module provides structured logging for audit purposes,
tracking API calls, risk assessments, and system events.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from src.utils.logging_config import setup_logging

logger = setup_logging("audit_logger")


class EventType(Enum):
    """Types of audit events."""
    API_CALL = "api_call"
    SAFETY_CHECK = "safety_check"
    RISK_ASSESSMENT = "risk_assessment"
    CRISIS_INTERVENTION = "crisis_intervention"
    PII_REDACTION = "pii_redaction"
    USER_INPUT = "user_input"
    SYSTEM_RESPONSE = "system_response"
    ERROR = "error"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


@dataclass
class AuditEvent:
    """An audit event record."""
    event_type: EventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["event_type"] = self.event_type.value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class AuditLoggerConfig:
    """Configuration for audit logger."""
    log_dir: str = "logs/audit"
    max_file_size_mb: int = 10
    retention_days: int = 30
    log_to_console: bool = False
    log_pii: bool = False  # Never log actual PII, only counts
    enabled: bool = True


class AuditLogger:
    """
    Structured audit logger for system events.

    Provides methods for logging API calls, safety checks,
    and other events for compliance and debugging.

    Example:
        audit = AuditLogger()
        audit.log_api_call(session_id="123", endpoint="/chat", status=200)
    """

    def __init__(self, config: Optional[AuditLoggerConfig] = None):
        """
        Initialize audit logger.

        Args:
            config: Logger configuration
        """
        self.config = config or AuditLoggerConfig()
        self._events: List[AuditEvent] = []
        self._current_file = None

        if self.config.enabled:
            self._ensure_log_dir()

        logger.info("AuditLogger initialized")

    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def _get_log_file_path(self) -> str:
        """Get current log file path."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.config.log_dir, f"audit_{date_str}.jsonl")

    def log(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: The event to log
        """
        if not self.config.enabled:
            return

        self._events.append(event)

        # Write to file
        try:
            log_path = self._get_log_file_path()
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

        if self.config.log_to_console:
            logger.info(f"AUDIT: {event.event_type.value} - {event.data}")

    def log_api_call(
        self,
        session_id: str,
        endpoint: str,
        status: int,
        latency_ms: float,
        tokens_used: int = 0,
        error: Optional[str] = None
    ) -> None:
        """Log an API call event."""
        event = AuditEvent(
            event_type=EventType.API_CALL,
            session_id=session_id,
            data={
                "endpoint": endpoint,
                "status": status,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "error": error
            }
        )
        self.log(event)

    def log_safety_check(
        self,
        session_id: str,
        risk_level: str,
        is_safe: bool,
        matched_pattern: Optional[str] = None,
        action_taken: Optional[str] = None
    ) -> None:
        """Log a safety check event."""
        event = AuditEvent(
            event_type=EventType.SAFETY_CHECK,
            session_id=session_id,
            data={
                "risk_level": risk_level,
                "is_safe": is_safe,
                "matched_pattern": matched_pattern,
                "action_taken": action_taken
            }
        )
        self.log(event)

    def log_risk_assessment(
        self,
        session_id: str,
        risk_level: str,
        primary_concern: str,
        approach: str,
        key_points: List[str]
    ) -> None:
        """Log a risk assessment event."""
        event = AuditEvent(
            event_type=EventType.RISK_ASSESSMENT,
            session_id=session_id,
            data={
                "risk_level": risk_level,
                "primary_concern": primary_concern,
                "approach": approach,
                "key_points": key_points
            }
        )
        self.log(event)

    def log_crisis_intervention(
        self,
        session_id: str,
        trigger: str,
        resources_provided: List[str],
        escalated: bool
    ) -> None:
        """Log a crisis intervention event."""
        event = AuditEvent(
            event_type=EventType.CRISIS_INTERVENTION,
            session_id=session_id,
            data={
                "trigger": trigger,
                "resources_provided": resources_provided,
                "escalated": escalated
            }
        )
        self.log(event)

    def log_pii_redaction(
        self,
        session_id: str,
        entity_count: int,
        entity_types: List[str]
    ) -> None:
        """Log PII redaction event (counts only, not actual PII)."""
        event = AuditEvent(
            event_type=EventType.PII_REDACTION,
            session_id=session_id,
            data={
                "entity_count": entity_count,
                "entity_types": entity_types
            }
        )
        self.log(event)

    def log_error(
        self,
        session_id: Optional[str],
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None
    ) -> None:
        """Log an error event."""
        event = AuditEvent(
            event_type=EventType.ERROR,
            session_id=session_id,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "stack_trace": stack_trace
            }
        )
        self.log(event)

    def log_session_start(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log session start event."""
        event = AuditEvent(
            event_type=EventType.SESSION_START,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        self.log(event)

    def log_session_end(
        self,
        session_id: str,
        duration_seconds: float,
        turn_count: int
    ) -> None:
        """Log session end event."""
        event = AuditEvent(
            event_type=EventType.SESSION_END,
            session_id=session_id,
            data={
                "duration_seconds": duration_seconds,
                "turn_count": turn_count
            }
        )
        self.log(event)

    def get_recent_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Get recent events (from memory)."""
        events = self._events

        if session_id:
            events = [e for e in events if e.session_id == session_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of events for a session."""
        events = [e for e in self._events if e.session_id == session_id]

        return {
            "session_id": session_id,
            "event_count": len(events),
            "event_types": list(set(e.event_type.value for e in events)),
            "safety_checks": sum(1 for e in events if e.event_type == EventType.SAFETY_CHECK),
            "api_calls": sum(1 for e in events if e.event_type == EventType.API_CALL),
            "errors": sum(1 for e in events if e.event_type == EventType.ERROR)
        }
