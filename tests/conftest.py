"""
Pytest configuration and fixtures.

This module provides shared fixtures for all tests,
ensuring MOCK mode is used throughout.
"""

import os
import pytest
import asyncio

# CRITICAL: Set MOCK mode before any imports
os.environ["LLM_TYPE"] = "MOCK"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mode():
    """Ensure mock mode is set."""
    original = os.environ.get("LLM_TYPE")
    os.environ["LLM_TYPE"] = "MOCK"
    yield True
    if original:
        os.environ["LLM_TYPE"] = original


@pytest.fixture
async def mock_agent():
    """Create a mock-mode PsychologistAgent."""
    from src.main import PsychologistAgent

    agent = PsychologistAgent(mock_mode=True)
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.fixture
async def mock_session(mock_agent):
    """Create a mock session."""
    session = await mock_agent.session_manager.create_session()
    yield session
    await mock_agent.session_manager.end_session(session.session_id)


@pytest.fixture
def safety_gateway():
    """Create a mock-mode SafetyGateway."""
    from src.safety.gateway import SafetyGateway
    from src.safety.embeddings import EmbeddingManager
    from src.safety.patterns import PatternLoader

    EmbeddingManager.reset_instance()
    PatternLoader.clear_cache()

    gateway = SafetyGateway(mock_mode=True)
    yield gateway

    EmbeddingManager.reset_instance()
    PatternLoader.clear_cache()


@pytest.fixture
def pii_redactor():
    """Create a mock-mode PIIRedactor."""
    from src.privacy.pii_redactor import PIIRedactor
    return PIIRedactor(mock_mode=True, use_presidio=False)


@pytest.fixture
async def rag_retriever():
    """Create a mock-mode RAGRetriever."""
    from src.rag.retriever import RAGRetriever
    from src.safety.embeddings import EmbeddingManager

    EmbeddingManager.reset_instance()

    retriever = RAGRetriever(mock_mode=True)
    await retriever.initialize()
    yield retriever

    EmbeddingManager.reset_instance()


@pytest.fixture
def prompt_generator():
    """Create a PromptGenerator."""
    from src.prompt.generator import PromptGenerator
    return PromptGenerator()


@pytest.fixture
def deepseek_client():
    """Create a mock-mode DeepseekClient."""
    from src.api.deepseek_client import DeepseekClient
    return DeepseekClient(mock_mode=True)


@pytest.fixture
def risk_checker():
    """Create a RiskChecker."""
    from src.audit.risk_checker import RiskChecker
    return RiskChecker()


@pytest.fixture
def crisis_handler():
    """Create a CrisisHandler."""
    from src.audit.crisis_handler import CrisisHandler
    return CrisisHandler()


@pytest.fixture
async def local_generator():
    """Create a mock-mode LocalGenerator."""
    from src.inference.generator import LocalGenerator

    generator = LocalGenerator(mock_mode=True)
    await generator.initialize()
    yield generator
    await generator.unload()


@pytest.fixture
def memory_store():
    """Create a MemoryStore."""
    from src.memory.store import MemoryStore
    return MemoryStore()


@pytest.fixture
def session_manager(memory_store):
    """Create a SessionManager."""
    from src.session.manager import SessionManager
    return SessionManager(memory_store=memory_store)


@pytest.fixture
def audit_logger(tmp_path):
    """Create an AuditLogger with temp directory."""
    from src.audit.logger import AuditLogger, AuditLoggerConfig

    config = AuditLoggerConfig(
        log_dir=str(tmp_path / "audit"),
        log_to_console=False
    )
    return AuditLogger(config)


# Helper functions for tests
def assert_safe_response(result):
    """Assert that a response is safe (no crisis indicators)."""
    assert result.get("requires_crisis_response", False) is False
    assert result.get("risk_level", "none") in ["none", "low"]


def assert_crisis_response(result):
    """Assert that a response indicates crisis."""
    assert result.get("requires_crisis_response", False) is True
    assert "988" in result.get("response", "") or result.get("risk_level") in ["high", "critical"]
