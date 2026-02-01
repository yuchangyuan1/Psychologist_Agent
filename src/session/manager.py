"""
Session Manager for user session handling.

This module provides the SessionManager class for managing
user sessions, including creation, retrieval, and cleanup.
"""

import os
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from src.memory.store import MemoryStore, MemoryConfig
from src.utils.logging_config import setup_logging

logger = setup_logging("session_manager")


@dataclass
class SessionData:
    """Data associated with a session."""
    session_id: str
    user_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    risk_level: str = "none"
    turn_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "risk_level": self.risk_level,
            "turn_count": self.turn_count,
            "metadata": self.metadata
        }


@dataclass
class SessionManagerConfig:
    """Configuration for session manager."""
    session_timeout_minutes: int = 60
    max_sessions: int = 1000
    cleanup_interval_minutes: int = 15
    persist_sessions: bool = False
    persist_path: Optional[str] = None


class SessionManager:
    """
    Manager for user sessions.

    Handles session lifecycle including creation, retrieval,
    updates, and cleanup of stale sessions.

    Example:
        manager = SessionManager()
        session = await manager.create_session()
        # ... use session.session_id for conversation
        await manager.update_activity(session.session_id)
    """

    def __init__(
        self,
        config: Optional[SessionManagerConfig] = None,
        memory_store: Optional[MemoryStore] = None
    ):
        """
        Initialize session manager.

        Args:
            config: Session manager configuration
            memory_store: Memory store for conversation history
        """
        self.config = config or SessionManagerConfig()
        self.memory_store = memory_store or MemoryStore()
        self._sessions: Dict[str, SessionData] = {}

        logger.info("SessionManager initialized")

    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """
        Create a new session.

        Args:
            user_id: Optional user identifier
            metadata: Optional session metadata

        Returns:
            SessionData: Created session
        """
        # Cleanup old sessions if at capacity
        if len(self._sessions) >= self.config.max_sessions:
            await self._cleanup_old_sessions()

        session_id = str(uuid.uuid4())
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )

        self._sessions[session_id] = session
        logger.info(f"Created session: {session_id}")

        return session

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionData or None if not found
        """
        return self._sessions.get(session_id)

    async def update_activity(
        self,
        session_id: str,
        risk_level: Optional[str] = None
    ) -> bool:
        """
        Update session activity timestamp.

        Args:
            session_id: Session identifier
            risk_level: Optional updated risk level

        Returns:
            bool: True if session was updated
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.last_activity = datetime.now().isoformat()
        session.turn_count += 1

        if risk_level:
            session.risk_level = risk_level

        return True

    async def set_metadata(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> bool:
        """
        Set session metadata.

        Args:
            session_id: Session identifier
            key: Metadata key
            value: Metadata value

        Returns:
            bool: True if successful
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.metadata[key] = value
        return True

    async def get_metadata(
        self,
        session_id: str,
        key: Optional[str] = None
    ) -> Any:
        """
        Get session metadata.

        Args:
            session_id: Session identifier
            key: Optional specific key to retrieve

        Returns:
            Metadata value or dict
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        if key:
            return session.metadata.get(key)
        return dict(session.metadata)

    async def end_session(self, session_id: str) -> bool:
        """
        End and remove a session.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session was ended
        """
        if session_id not in self._sessions:
            return False

        # Clear memory for session
        await self.memory_store.clear_session(session_id)

        del self._sessions[session_id]
        logger.info(f"Ended session: {session_id}")

        return True

    async def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List of message dictionaries
        """
        messages = await self.memory_store.get_history(session_id, limit)
        return [{"role": m.role, "content": m.content} for m in messages]

    async def add_to_history(
        self,
        session_id: str,
        user_input: str,
        response: str
    ) -> None:
        """
        Add a conversation turn to history.

        Args:
            session_id: Session identifier
            user_input: User's message
            response: Agent's response
        """
        await self.memory_store.add(session_id, user_input, response)
        await self.update_activity(session_id)

    async def get_all_sessions(self) -> List[SessionData]:
        """Get all active sessions."""
        return list(self._sessions.values())

    async def get_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._sessions)

    async def _cleanup_old_sessions(self) -> int:
        """
        Remove sessions that have timed out.

        Returns:
            int: Number of sessions removed
        """
        timeout = timedelta(minutes=self.config.session_timeout_minutes)
        now = datetime.now()
        removed = 0

        sessions_to_remove = []
        for session_id, session in self._sessions.items():
            last_activity = datetime.fromisoformat(session.last_activity)
            if now - last_activity > timeout:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            await self.end_session(session_id)
            removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} stale sessions")

        return removed

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Session statistics dictionary
        """
        session = self._sessions.get(session_id)
        if not session:
            return {}

        memory_stats = await self.memory_store.get_session_stats(session_id)

        return {
            **session.to_dict(),
            "memory": memory_stats
        }
