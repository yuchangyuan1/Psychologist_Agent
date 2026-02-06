"""
Gradio Demo UI for Psychologist Agent.

This module provides a web-based chat interface for interacting
with the Psychologist Agent.
"""

import os
import asyncio
from typing import List, Dict

# Set mock mode for demo
os.environ.setdefault("LLM_TYPE", "MOCK")

from src.main import PsychologistAgent
from src.utils.logging_config import setup_logging

logger = setup_logging("demo_app")


# Global agent instance
agent: PsychologistAgent = None
current_session_id: str = None


async def initialize_agent():
    """Initialize the agent."""
    global agent
    if agent is None:
        agent = PsychologistAgent()
        await agent.initialize()
        logger.info("Agent initialized for demo")


async def create_new_session():
    """Create a new chat session."""
    global current_session_id
    await initialize_agent()
    session = await agent.session_manager.create_session()
    current_session_id = session.session_id
    return current_session_id


async def chat(message: str, history: List[Dict[str, str]]):
    """
    Process a chat message.

    Args:
        message: User message
        history: Chat history (Gradio messages format)

    Returns:
        Updated history
    """
    global current_session_id

    await initialize_agent()

    if current_session_id is None:
        await create_new_session()

    try:
        result = await agent.process_message(message, current_session_id)
        response = result["response"]

        # Add crisis indicator if needed
        if result.get("requires_crisis_response"):
            response = f"[CRISIS SUPPORT]\n\n{response}"

    except Exception as e:
        logger.error(f"Chat error: {e}")
        response = "I apologize, but I encountered an error. If you're in crisis, please call 988 for immediate support."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history


def run_chat(message: str, history: List[Dict[str, str]]):
    """Synchronous wrapper for chat function."""
    return asyncio.run(chat(message, history))


def clear_chat():
    """Clear chat and create new session."""
    global current_session_id
    current_session_id = None
    asyncio.run(create_new_session())
    return []


def create_demo():
    """Create and return the Gradio demo interface."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio is required for the demo. Install with: pip install gradio")

    with gr.Blocks(title="Psychologist Agent Demo") as demo:
        gr.Markdown("""
        # Psychologist Agent Demo

        A mental health support assistant powered by AI.

        **Disclaimer**: This is a demonstration system for educational purposes.
        It is not a substitute for professional mental health care.
        If you're experiencing a mental health crisis, please contact:
        - **988 Suicide & Crisis Lifeline**: Call or text 988
        - **Crisis Text Line**: Text HOME to 741741
        - **Emergency**: Call 911

        ---
        """)

        chatbot = gr.Chatbot(
            label="Conversation",
            height=500
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                scale=4
            )
            submit = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear = gr.Button("Clear Chat")
            new_session = gr.Button("New Session")

        with gr.Accordion("About this demo", open=False):
            gr.Markdown("""
            ### How it works

            This demo uses a multi-stage pipeline:

            1. **Safety Gateway**: Screens messages for crisis indicators
            2. **PII Redaction**: Removes personal information before processing
            3. **RAG Retrieval**: Retrieves relevant therapeutic knowledge
            4. **Cloud Analysis**: Analyzes the message for clinical insights
            5. **Risk Audit**: Validates the analysis and checks for risks
            6. **Local Generation**: Generates a supportive response

            ### Features

            - CBT and DBT therapeutic techniques
            - Crisis detection and intervention
            - Privacy-preserving processing
            - Conversation memory

            ### Limitations

            - This is a demo with mock responses in MOCK mode
            - Not a replacement for professional mental health care
            - May not always provide appropriate responses
            """)

        # Event handlers
        submit.click(
            run_chat,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        msg.submit(
            run_chat,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        clear.click(
            clear_chat,
            outputs=[chatbot]
        )

        new_session.click(
            clear_chat,
            outputs=[chatbot]
        )

    return demo


def main():
    """Run the demo."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
