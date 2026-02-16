"""
Psychologist Agent - Main Orchestrator.

This module provides the main PsychologistAgent class that orchestrates
the complete inference pipeline from user input to response generation.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass

from src.safety.gateway import SafetyGateway, SafetyResult
from src.safety.patterns import RiskLevel
from src.privacy.pii_redactor import PIIRedactor, RedactionResult
from src.rag.retriever import RAGRetriever, ContextBuilder
from src.prompt.generator import PromptGenerator, PromptConfig
from src.api.deepseek_client import DeepseekClient
from src.api.models import AnalysisResult
from src.audit.risk_checker import RiskChecker, RiskAssessment
from src.audit.crisis_handler import CrisisHandler, CrisisResponse
from src.audit.logger import AuditLogger, AuditLoggerConfig
from src.inference.generator import LocalGenerator
from src.memory.store import MemoryStore
from src.session.manager import SessionManager
from src.utils.logging_config import setup_logging

logger = setup_logging("psychologist_agent")


@dataclass
class AgentConfig:
    """Configuration for PsychologistAgent."""
    enable_safety_check: bool = True
    enable_pii_redaction: bool = True
    enable_rag: bool = True
    enable_cloud_analysis: bool = True
    enable_risk_audit: bool = True
    enable_audit_logging: bool = True
    max_cloud_history_turns: int = 10
    max_local_history_turns: int = 3


class PsychologistAgent:
    """
    Main orchestrator for the Psychologist Agent.

    Implements the complete inference pipeline:
    User Input → Safety Gateway → PII Redaction → RAG Retrieval
        → Cloud Analysis (Deepseek) → Risk Audit
        → Local Generation (GGUF) → Memory Update → Response

    Example:
        agent = PsychologistAgent()
        await agent.initialize()
        result = await agent.process_message("I'm feeling anxious", session_id)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        mock_mode: Optional[bool] = None
    ):
        """
        Initialize the Psychologist Agent.

        Args:
            config: Agent configuration
            mock_mode: Whether to use mock mode for all components
        """
        self.config = config or AgentConfig()
        self.mock_mode = mock_mode
        if self.mock_mode is None:
            self.mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"

        # Initialize components
        # Safety and RAG always use real embeddings (BGE-small, CPU)
        # for meaningful semantic matching, even in MOCK mode
        self.safety_gateway = SafetyGateway(mock_mode=False)
        self.pii_redactor = PIIRedactor(mock_mode=self.mock_mode)
        self.rag_retriever = RAGRetriever(mock_mode=False)
        self.prompt_generator = PromptGenerator()
        self.cloud_client = DeepseekClient(mock_mode=self.mock_mode)
        self.risk_checker = RiskChecker()
        self.crisis_handler = CrisisHandler()
        self.local_generator = LocalGenerator(mock_mode=self.mock_mode)
        self.memory_store = MemoryStore()
        self.session_manager = SessionManager(memory_store=self.memory_store)

        if self.config.enable_audit_logging:
            self.audit_logger = AuditLogger()
        else:
            self.audit_logger = None

        self._initialized = False

        logger.info(f"PsychologistAgent created (mock_mode={self.mock_mode})")

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing PsychologistAgent...")

        # Initialize RAG (loads knowledge base)
        if self.config.enable_rag:
            await self.rag_retriever.initialize()

        # Initialize local generator (loads model)
        await self.local_generator.initialize()

        self._initialized = True
        logger.info("PsychologistAgent initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        if self.local_generator:
            await self.local_generator.unload()

        if self.cloud_client:
            await self.cloud_client.close()

        self._initialized = False
        logger.info("PsychologistAgent shutdown complete")

    async def process_message(
        self,
        user_input: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Process a user message through the complete pipeline.

        Args:
            user_input: User's message
            session_id: Session identifier

        Returns:
            Dict containing response and metadata
        """
        if not self._initialized:
            await self.initialize()

        result = {
            "response": "",
            "risk_level": "none",
            "requires_crisis_response": False,
            "session_id": session_id,
            "pipeline_details": {}
        }

        try:
            # Step 1: Safety Gateway Check
            if self.config.enable_safety_check:
                safety_result = await self.safety_gateway.check(user_input)

                result["pipeline_details"]["safety"] = {
                    "is_safe": safety_result.is_safe,
                    "risk_level": safety_result.risk_level.value,
                    "matched_pattern": safety_result.matched_pattern,
                    "matched_category": safety_result.matched_category,
                    "similarity_score": round(safety_result.similarity_score, 4),
                    "action": safety_result.action
                }

                if self.audit_logger:
                    self.audit_logger.log_safety_check(
                        session_id=session_id,
                        risk_level=safety_result.risk_level.value,
                        is_safe=safety_result.is_safe,
                        matched_pattern=safety_result.matched_pattern,
                        action_taken=safety_result.action
                    )

                # Handle immediate crisis
                if not safety_result.is_safe:
                    result["response"] = safety_result.response
                    result["risk_level"] = safety_result.risk_level.value
                    result["requires_crisis_response"] = True
                    result["resources"] = safety_result.resources

                    # Still save to history
                    await self.session_manager.add_to_history(
                        session_id, user_input, safety_result.response
                    )
                    await self.session_manager.update_activity(
                        session_id, risk_level=safety_result.risk_level.value
                    )

                    return result

            # Step 2: PII Redaction
            if self.config.enable_pii_redaction:
                redaction_result = self.pii_redactor.redact(user_input)
                sanitized_input = redaction_result.redacted_text

                result["pipeline_details"]["pii"] = {
                    "entity_count": redaction_result.entity_count,
                    "entities": [
                        {"type": e.entity_type.value, "replacement": e.replacement}
                        for e in redaction_result.entities
                    ],
                    "redacted_text": redaction_result.redacted_text
                }

                if self.audit_logger and redaction_result.entity_count > 0:
                    self.audit_logger.log_pii_redaction(
                        session_id=session_id,
                        entity_count=redaction_result.entity_count,
                        entity_types=[e.entity_type.value for e in redaction_result.entities]
                    )
            else:
                sanitized_input = user_input

            # Step 3: RAG Retrieval
            rag_context = ""
            if self.config.enable_rag:
                rag_results = await self.rag_retriever.retrieve(sanitized_input)
                rag_context = self.rag_retriever.format_context(rag_results)

                result["pipeline_details"]["rag"] = {
                    "num_chunks": len(rag_results),
                    "chunks": [
                        {
                            "source": r.source,
                            "source_type": r.source_type,
                            "score": round(r.score, 4),
                            "text_preview": r.content[:150] + "..."
                            if len(r.content) > 150
                            else r.content
                        }
                        for r in rag_results[:3]
                    ]
                }

            # Step 4: Get conversation history (separate for cloud and local)
            cloud_history, user_profile = await self.memory_store.get_cloud_context(session_id)

            # Step 5: Cloud Analysis (Deepseek) with profile
            if self.config.enable_cloud_analysis:
                cloud_prompt = self.prompt_generator.gen_cloud_prompt(
                    sanitized_input=sanitized_input,
                    rag_context=rag_context,
                    history=cloud_history,
                    user_profile=user_profile.to_json() if user_profile else None
                )

                cloud_analysis = await self.cloud_client.analyze(
                    system_message=cloud_prompt.system_message,
                    user_message=cloud_prompt.user_message
                )

                result["pipeline_details"]["cloud_analysis"] = {
                    "risk_level": cloud_analysis.risk_level.value,
                    "primary_concern": cloud_analysis.primary_concern,
                    "suggested_approach": cloud_analysis.suggested_approach.value,
                    "suggested_technique": cloud_analysis.suggested_technique,
                    "guidance": cloud_analysis.guidance_for_local_model,
                    "key_points": cloud_analysis.key_points
                }

                # Update profile if cloud analysis provides updates
                if cloud_analysis.updated_user_profile:
                    await self.memory_store.update_profile(
                        session_id, cloud_analysis.updated_user_profile
                    )
                    result["pipeline_details"]["profile_update"] = cloud_analysis.updated_user_profile
            else:
                # Default analysis if cloud disabled
                cloud_analysis = AnalysisResult()

            # Step 6: Risk Audit
            if self.config.enable_risk_audit:
                risk_assessment = self.risk_checker.assess(
                    cloud_analysis, user_input
                )

                if self.audit_logger:
                    self.audit_logger.log_risk_assessment(
                        session_id=session_id,
                        risk_level=risk_assessment.risk_level.value,
                        primary_concern=cloud_analysis.primary_concern,
                        approach=cloud_analysis.suggested_approach.value,
                        key_points=cloud_analysis.key_points
                    )

                result["pipeline_details"]["risk_audit"] = {
                    "risk_level": risk_assessment.risk_level.value,
                    "requires_crisis": risk_assessment.requires_crisis_response,
                    "recommended_actions": getattr(risk_assessment, "recommended_actions", [])
                }

                # Handle crisis from risk audit
                if risk_assessment.requires_crisis_response:
                    crisis_response = self.crisis_handler.get_response(risk_assessment)
                    result["response"] = crisis_response.message
                    result["risk_level"] = risk_assessment.risk_level.value
                    result["requires_crisis_response"] = True

                    if self.audit_logger:
                        self.audit_logger.log_crisis_intervention(
                            session_id=session_id,
                            trigger=crisis_response.response_type,
                            resources_provided=[r.name for r in crisis_response.resources],
                            escalated=crisis_response.requires_escalation
                        )

                    await self.session_manager.add_to_history(
                        session_id, user_input, crisis_response.message
                    )

                    return result

                result["risk_level"] = risk_assessment.risk_level.value

            # Step 7: Local Generation (GGUF) with 3-turn history and messages list
            local_history = await self.memory_store.get_local_context(session_id)

            local_prompt = self.prompt_generator.gen_local_prompt(
                user_input=user_input,
                cloud_analysis=cloud_analysis.to_dict(),
                rag_context=rag_context,
                history=local_history
            )

            # Use create_chat_completion with messages list
            generation_result = await self.local_generator.create_chat_completion(
                messages=local_prompt.to_messages()
            )

            result["response"] = generation_result.text

            # Step 8: Update memory
            await self.session_manager.add_to_history(
                session_id, user_input, generation_result.text
            )

            return result

        except Exception as e:
            logger.error(f"Error processing message: {e}")

            if self.audit_logger:
                self.audit_logger.log_error(
                    session_id=session_id,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )

            result["response"] = "I apologize, but I'm having trouble processing your message right now. If you're in crisis, please call 988 for immediate support."
            result["error"] = str(e)

            return result

    async def process_message_stream(
        self,
        user_input: str,
        session_id: str
    ) -> AsyncIterator[str]:
        """
        Process message with streaming response.

        Args:
            user_input: User's message
            session_id: Session identifier

        Yields:
            Response tokens
        """
        if not self._initialized:
            await self.initialize()

        # Safety check first
        if self.config.enable_safety_check:
            safety_result = await self.safety_gateway.check(user_input)
            if not safety_result.is_safe:
                yield safety_result.response
                return

        # PII redaction
        if self.config.enable_pii_redaction:
            redaction_result = self.pii_redactor.redact(user_input)
            sanitized_input = redaction_result.redacted_text
        else:
            sanitized_input = user_input

        # RAG retrieval
        rag_context = ""
        if self.config.enable_rag:
            rag_results = await self.rag_retriever.retrieve(sanitized_input)
            rag_context = self.rag_retriever.format_context(rag_results)

        # Get cloud context (10 turns + profile)
        cloud_history, user_profile = await self.memory_store.get_cloud_context(session_id)

        # Cloud analysis with profile
        if self.config.enable_cloud_analysis:
            cloud_prompt = self.prompt_generator.gen_cloud_prompt(
                sanitized_input=sanitized_input,
                rag_context=rag_context,
                history=cloud_history,
                user_profile=user_profile.to_json() if user_profile else None
            )
            cloud_analysis = await self.cloud_client.analyze(
                system_message=cloud_prompt.system_message,
                user_message=cloud_prompt.user_message
            )

            # Update profile if cloud analysis provides updates
            if cloud_analysis.updated_user_profile:
                await self.memory_store.update_profile(
                    session_id, cloud_analysis.updated_user_profile
                )
        else:
            cloud_analysis = AnalysisResult()

        # Risk audit
        if self.config.enable_risk_audit:
            risk_assessment = self.risk_checker.assess(cloud_analysis, user_input)
            if risk_assessment.requires_crisis_response:
                crisis_response = self.crisis_handler.get_response(risk_assessment)
                yield crisis_response.message
                return

        # Get local context (3 turns only)
        local_history = await self.memory_store.get_local_context(session_id)

        # Generate with streaming
        local_prompt = self.prompt_generator.gen_local_prompt(
            user_input=user_input,
            cloud_analysis=cloud_analysis.to_dict(),
            rag_context=rag_context,
            history=local_history
        )

        full_response = ""
        async for token in self.local_generator.generate_stream(
            local_prompt.full_prompt
        ):
            full_response += token
            yield token

        # Save to history
        await self.session_manager.add_to_history(
            session_id, user_input, full_response
        )


async def main():
    """Main entry point for running the agent."""
    agent = PsychologistAgent()
    await agent.initialize()

    # Create a session
    session = await agent.session_manager.create_session()
    print(f"Created session: {session.session_id}")

    # Example conversation
    messages = [
        "Hi, I've been feeling really anxious lately about work.",
        "It's hard to concentrate and I feel overwhelmed.",
        "What can I do to feel better?"
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        result = await agent.process_message(msg, session.session_id)
        print(f"Agent: {result['response']}")
        print(f"Risk Level: {result['risk_level']}")

    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
