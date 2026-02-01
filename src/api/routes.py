"""
FastAPI routes for the Psychologist Agent API.

This module provides the HTTP endpoints for the agent,
including chat, session management, and health checks.
"""

import os
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from src.utils.logging_config import setup_logging

logger = setup_logging("api_routes")


# Pydantic models
try:
    from pydantic import BaseModel, Field

    class ChatRequest(BaseModel):
        """Chat request model."""
        message: str = Field(..., description="User message")
        session_id: Optional[str] = Field(None, description="Session ID")

    class ChatResponse(BaseModel):
        """Chat response model."""
        response: str
        session_id: str
        risk_level: str
        requires_crisis_response: bool

    class SessionResponse(BaseModel):
        """Session info response."""
        session_id: str
        created_at: str
        turn_count: int
        risk_level: str

    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        version: str
        mode: str

except ImportError:
    ChatRequest = dict
    ChatResponse = dict
    SessionResponse = dict
    HealthResponse = dict


def create_api_routes(agent):
    """
    Create FastAPI router with agent endpoints.

    Args:
        agent: PsychologistAgent instance

    Returns:
        FastAPI router
    """
    try:
        from fastapi import APIRouter, HTTPException, Depends
        from fastapi.responses import StreamingResponse
    except ImportError:
        logger.error("fastapi not installed")
        raise ImportError("fastapi is required")

    router = APIRouter(prefix="/api/v1", tags=["chat"])

    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "mode": "MOCK" if agent.mock_mode else "PRODUCTION"
        }

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Process a chat message.

        Creates a new session if none provided.
        """
        try:
            # Get or create session
            session_id = request.session_id
            if not session_id:
                session = await agent.session_manager.create_session()
                session_id = session.session_id

            # Process message
            result = await agent.process_message(
                user_input=request.message,
                session_id=session_id
            )

            return {
                "response": result["response"],
                "session_id": session_id,
                "risk_level": result.get("risk_level", "none"),
                "requires_crisis_response": result.get("requires_crisis_response", False)
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """
        Process a chat message with streaming response.
        """
        try:
            session_id = request.session_id
            if not session_id:
                session = await agent.session_manager.create_session()
                session_id = session.session_id

            async def stream_generator():
                async for token in agent.process_message_stream(
                    user_input=request.message,
                    session_id=session_id
                ):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )

        except Exception as e:
            logger.error(f"Stream error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/session/create", response_model=SessionResponse)
    async def create_session():
        """Create a new session."""
        session = await agent.session_manager.create_session()
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "turn_count": 0,
            "risk_level": "none"
        }

    @router.get("/session/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        """Get session information."""
        session = await agent.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "turn_count": session.turn_count,
            "risk_level": session.risk_level
        }

    @router.delete("/session/{session_id}")
    async def end_session(session_id: str):
        """End a session."""
        success = await agent.session_manager.end_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ended", "session_id": session_id}

    @router.get("/session/{session_id}/history")
    async def get_history(session_id: str, limit: int = 20):
        """Get session conversation history."""
        session = await agent.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        history = await agent.session_manager.get_session_history(
            session_id, limit=limit
        )
        return {"session_id": session_id, "history": history}

    @router.get("/crisis/resources")
    async def get_crisis_resources():
        """Get crisis resources."""
        return {
            "resources": [
                {
                    "name": "988 Suicide & Crisis Lifeline",
                    "phone": "988",
                    "description": "24/7 crisis support"
                },
                {
                    "name": "Crisis Text Line",
                    "text": "HOME to 741741",
                    "description": "Text-based crisis support"
                }
            ]
        }

    return router


def create_app(agent=None):
    """
    Create FastAPI application.

    Args:
        agent: Optional PsychologistAgent instance

    Returns:
        FastAPI application
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("fastapi is required")

    # Import here to avoid circular imports
    if agent is None:
        from src.main import PsychologistAgent
        mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"
        agent = PsychologistAgent(mock_mode=mock_mode)

    @asynccontextmanager
    async def lifespan(app):
        # Startup
        await agent.initialize()
        yield
        # Shutdown
        await agent.shutdown()

    app = FastAPI(
        title="Psychologist Agent API",
        description="AI-powered mental health support assistant",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routes
    router = create_api_routes(agent)
    app.include_router(router)

    return app
