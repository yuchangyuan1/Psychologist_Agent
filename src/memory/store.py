"""
Memory store for conversation history management.

This module provides the MemoryStore class for storing and retrieving
conversation history, supporting both in-memory and persistent storage.
Also provides UserProfile for long-term clinical context.
"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from src.utils.logging_config import setup_logging

logger = setup_logging("memory_store")


@dataclass
class UserProfile:
    """Long-term user profile for clinical context."""
    user_id: str = ""
    diagnosis_history: List[str] = field(default_factory=list)
    chronic_stressors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    therapeutic_preferences: Dict[str, Any] = field(default_factory=dict)
    previous_concerns: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create from JSON dictionary."""
        return cls(
            user_id=data.get("user_id", ""),
            diagnosis_history=data.get("diagnosis_history", []),
            chronic_stressors=data.get("chronic_stressors", []),
            risk_factors=data.get("risk_factors", []),
            therapeutic_preferences=data.get("therapeutic_preferences", {}),
            previous_concerns=data.get("previous_concerns", []),
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConversationSummary:
    """Summary of a conversation for context compression."""
    summary: str
    key_topics: List[str]
    emotional_themes: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MemoryConfig:
    """Configuration for memory store."""
    max_messages: int = 50
    summary_threshold: int = 20
    persist_path: Optional[str] = None
    auto_persist: bool = True


class MemoryStore:
    """
    Store for conversation history and context.

    Maintains per-session conversation history with support for
    summarization and persistence.

    Example:
        store = MemoryStore()
        await store.add("session_123", "I'm feeling anxious", "That sounds difficult...")
        history = await store.get_history("session_123")
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory store.

        Args:
            config: Memory store configuration
        """
        self.config = config or MemoryConfig()
        self._sessions: Dict[str, List[Message]] = defaultdict(list)
        self._summaries: Dict[str, List[ConversationSummary]] = defaultdict(list)
        self._metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._profiles: Dict[str, UserProfile] = {}

        if self.config.persist_path:
            self._load_persisted()

        logger.info("MemoryStore initialized")

    async def add(
        self,
        session_id: str,
        user_input: str,
        response: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        response_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to history.

        Args:
            session_id: Session identifier
            user_input: User's message
            response: Assistant's response
            user_metadata: Metadata for user message
            response_metadata: Metadata for response
        """
        user_msg = Message(
            role="user",
            content=user_input,
            metadata=user_metadata or {}
        )
        assistant_msg = Message(
            role="assistant",
            content=response,
            metadata=response_metadata or {}
        )

        self._sessions[session_id].append(user_msg)
        self._sessions[session_id].append(assistant_msg)

        # Truncate if exceeding max
        if len(self._sessions[session_id]) > self.config.max_messages * 2:
            self._sessions[session_id] = self._sessions[session_id][-self.config.max_messages * 2:]

        if self.config.auto_persist and self.config.persist_path:
            await self._persist_session(session_id)

        logger.debug(f"Added conversation turn to session {session_id}")

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> List[Message]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            include_metadata: Whether to include message metadata

        Returns:
            List of Message objects
        """
        messages = self._sessions.get(session_id, [])

        if limit:
            messages = messages[-limit:]

        if not include_metadata:
            # Create copies without metadata
            messages = [
                Message(role=m.role, content=m.content, timestamp=m.timestamp)
                for m in messages
            ]

        return messages

    async def get_formatted_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        format_type: str = "simple"
    ) -> str:
        """
        Get formatted conversation history as string.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages
            format_type: Format type ("simple", "detailed", "chat")

        Returns:
            Formatted history string
        """
        messages = await self.get_history(session_id, limit)

        if not messages:
            return ""

        if format_type == "simple":
            parts = []
            for msg in messages:
                role = "User" if msg.role == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            return "\n".join(parts)

        elif format_type == "detailed":
            parts = []
            for msg in messages:
                role = "User" if msg.role == "user" else "Assistant"
                parts.append(f"[{msg.timestamp}] {role}:\n{msg.content}")
            return "\n\n".join(parts)

        elif format_type == "chat":
            parts = []
            for msg in messages:
                parts.append({"role": msg.role, "content": msg.content})
            return json.dumps(parts, indent=2)

        return ""

    async def get_last_n_turns(
        self,
        session_id: str,
        n: int = 3
    ) -> List[Dict[str, str]]:
        """
        Get last N conversation turns.

        Args:
            session_id: Session identifier
            n: Number of turns (pairs of user/assistant messages)

        Returns:
            List of message dictionaries
        """
        messages = await self.get_history(session_id, limit=n * 2)
        return [{"role": m.role, "content": m.content} for m in messages]

    async def get_profile(self, session_id: str) -> UserProfile:
        """
        Get user profile for session.

        Args:
            session_id: Session identifier

        Returns:
            UserProfile for the session (creates new if doesn't exist)
        """
        if session_id not in self._profiles:
            self._profiles[session_id] = UserProfile(user_id=session_id)
            logger.debug(f"Created new profile for session {session_id}")
        return self._profiles[session_id]

    async def update_profile(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update user profile from DeepSeek analysis.

        Args:
            session_id: Session identifier
            updates: Dictionary of profile updates
        """
        profile = await self.get_profile(session_id)

        # Update list fields by appending unique items
        list_fields = ["diagnosis_history", "chronic_stressors", "risk_factors", "previous_concerns"]
        for field_name in list_fields:
            if field_name in updates:
                current_list = getattr(profile, field_name)
                new_items = updates[field_name]
                if isinstance(new_items, list):
                    for item in new_items:
                        if item and item not in current_list:
                            current_list.append(item)
                elif new_items and new_items not in current_list:
                    current_list.append(new_items)

        # Update dict fields by merging
        if "therapeutic_preferences" in updates:
            profile.therapeutic_preferences.update(updates["therapeutic_preferences"])

        # Update timestamp
        profile.last_updated = datetime.now().isoformat()

        logger.debug(f"Updated profile for session {session_id}")

    async def get_cloud_context(
        self,
        session_id: str
    ) -> Tuple[List[Dict[str, str]], UserProfile]:
        """
        Get context for cloud API (10 turns + profile).

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (history list, user profile)
        """
        history = await self.get_last_n_turns(session_id, n=10)
        profile = await self.get_profile(session_id)
        return history, profile

    async def get_local_context(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get context for local model (3 turns only).

        Args:
            session_id: Session identifier

        Returns:
            List of recent message dictionaries
        """
        return await self.get_last_n_turns(session_id, n=3)

    async def set_session_metadata(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> None:
        """Set session-level metadata."""
        self._metadata[session_id][key] = value

    async def get_session_metadata(
        self,
        session_id: str,
        key: Optional[str] = None
    ) -> Any:
        """Get session-level metadata."""
        if key:
            return self._metadata[session_id].get(key)
        return dict(self._metadata[session_id])

    async def clear_session(self, session_id: str) -> None:
        """Clear all data for a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._summaries:
            del self._summaries[session_id]
        if session_id in self._metadata:
            del self._metadata[session_id]
        if session_id in self._profiles:
            del self._profiles[session_id]

        logger.info(f"Cleared session {session_id}")

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        messages = self._sessions.get(session_id, [])
        return {
            "message_count": len(messages),
            "turn_count": len(messages) // 2,
            "first_message": messages[0].timestamp if messages else None,
            "last_message": messages[-1].timestamp if messages else None,
            "has_summaries": len(self._summaries.get(session_id, [])) > 0
        }

    async def get_all_sessions(self) -> List[str]:
        """Get all session IDs."""
        return list(self._sessions.keys())

    def _load_persisted(self) -> None:
        """Load persisted sessions from disk."""
        if not self.config.persist_path or not os.path.exists(self.config.persist_path):
            return

        try:
            persist_dir = Path(self.config.persist_path)
            for session_file in persist_dir.glob("*.json"):
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                session_id = session_file.stem
                self._sessions[session_id] = [
                    Message.from_dict(m) for m in data.get("messages", [])
                ]
                self._metadata[session_id] = data.get("metadata", {})

            logger.info(f"Loaded {len(self._sessions)} sessions from disk")
        except Exception as e:
            logger.error(f"Error loading persisted sessions: {e}")

    async def _persist_session(self, session_id: str) -> None:
        """Persist a session to disk."""
        if not self.config.persist_path:
            return

        try:
            persist_dir = Path(self.config.persist_path)
            persist_dir.mkdir(parents=True, exist_ok=True)

            session_file = persist_dir / f"{session_id}.json"
            data = {
                "messages": [m.to_dict() for m in self._sessions[session_id]],
                "metadata": self._metadata.get(session_id, {}),
                "updated_at": datetime.now().isoformat()
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error persisting session {session_id}: {e}")


class ConversationContext:
    """Helper for building conversation context from memory."""

    def __init__(self, memory_store: MemoryStore):
        self.store = memory_store

    async def build_context(
        self,
        session_id: str,
        max_turns: int = 5,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Build conversation context for model input.

        Args:
            session_id: Session identifier
            max_turns: Maximum conversation turns to include
            include_summary: Whether to include session summary

        Returns:
            Context dictionary
        """
        history = await self.store.get_last_n_turns(session_id, max_turns)

        context = {
            "conversation_history": history,
            "turn_count": len(history) // 2,
            "session_id": session_id
        }

        # Add session metadata
        metadata = await self.store.get_session_metadata(session_id)
        if metadata:
            context["session_metadata"] = metadata

        return context
