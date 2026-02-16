"""
Gradio Demo UI for Psychologist Agent.

This module provides a web-based chat interface for interacting
with the Psychologist Agent, with pipeline visibility for presentations.
"""

import os
import json
import asyncio
from typing import List, Dict, Any

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


def format_pipeline_details(result: Dict[str, Any]) -> str:
    """Format pipeline details as readable markdown for the debug panel."""
    details = result.get("pipeline_details", {})
    if not details:
        return "*No pipeline details available.*"

    sections = []

    # Safety Gateway
    safety = details.get("safety")
    if safety:
        status = "SAFE" if safety["is_safe"] else "BLOCKED"
        sections.append(
            f"### 1. Safety Gateway\n"
            f"- **Status**: {status}\n"
            f"- **Risk Level**: {safety['risk_level']}\n"
            f"- **Similarity Score**: {safety['similarity_score']}\n"
            f"- **Matched Pattern**: {safety.get('matched_pattern') or 'None'}\n"
            f"- **Category**: {safety.get('matched_category') or 'N/A'}\n"
            f"- **Action**: {safety.get('action') or 'pass'}"
        )

    # PII Redaction
    pii = details.get("pii")
    if pii:
        entity_info = ""
        if pii["entity_count"] > 0:
            entity_list = ", ".join(
                f"`{e['type']}` -> `{e['replacement']}`"
                for e in pii["entities"]
            )
            entity_info = f"\n- **Entities**: {entity_list}"
            entity_info += f"\n- **Sent to Cloud**: `{pii['redacted_text']}`"
        sections.append(
            f"### 2. PII Redaction\n"
            f"- **PII Found**: {pii['entity_count']}{entity_info}"
        )

    # RAG Retrieval
    rag = details.get("rag")
    if rag:
        chunks_info = ""
        if rag["chunks"]:
            chunk_lines = []
            for i, c in enumerate(rag["chunks"], 1):
                chunk_lines.append(
                    f"  {i}. [{c['source_type']}] score={c['score']} - "
                    f"{c['text_preview'][:80]}..."
                )
            chunks_info = "\n" + "\n".join(chunk_lines)
        sections.append(
            f"### 3. RAG Retrieval\n"
            f"- **Chunks Retrieved**: {rag['num_chunks']}"
            f"{chunks_info}"
        )

    # Cloud Analysis
    cloud = details.get("cloud_analysis")
    if cloud:
        key_points = ", ".join(cloud.get("key_points", [])) or "N/A"
        sections.append(
            f"### 4. Cloud Analysis (Deepseek Supervisor)\n"
            f"- **Risk Level**: {cloud['risk_level']}\n"
            f"- **Primary Concern**: {cloud.get('primary_concern') or 'N/A'}\n"
            f"- **Approach**: {cloud['suggested_approach']}\n"
            f"- **Technique**: {cloud.get('suggested_technique') or 'N/A'}\n"
            f"- **Guidance for Local**: {cloud.get('guidance') or 'N/A'}"
        )

    # Profile Update
    profile = details.get("profile_update")
    if profile:
        sections.append(
            f"### 4b. User Profile Update\n"
            f"```json\n{json.dumps(profile, indent=2)}\n```"
        )

    # Risk Audit
    risk = details.get("risk_audit")
    if risk:
        actions = ", ".join(risk.get("recommended_actions", [])) or "N/A"
        sections.append(
            f"### 5. Risk Audit\n"
            f"- **Final Risk Level**: {risk['risk_level']}\n"
            f"- **Crisis Required**: {risk['requires_crisis']}\n"
            f"- **Actions**: {actions}"
        )

    # Overall result
    sections.append(
        f"### 6. Final Output\n"
        f"- **Risk Level**: {result['risk_level']}\n"
        f"- **Crisis Response**: {result.get('requires_crisis_response', False)}\n"
        f"- **Response Length**: {len(result.get('response', ''))} chars"
    )

    return "\n\n---\n\n".join(sections)


async def chat(message: str, history: List[Dict[str, str]]):
    """
    Process a chat message.

    Args:
        message: User message
        history: Chat history (Gradio messages format)

    Returns:
        Tuple of (updated history, pipeline details markdown)
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

        pipeline_md = format_pipeline_details(result)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        response = "I apologize, but I encountered an error. If you're in crisis, please call 988 for immediate support."
        pipeline_md = f"**Error**: {str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, pipeline_md


def run_chat(message: str, history: List[Dict[str, str]]):
    """Synchronous wrapper for chat function."""
    return asyncio.run(chat(message, history))


def clear_chat():
    """Clear chat and create new session."""
    global current_session_id
    current_session_id = None
    asyncio.run(create_new_session())
    return [], "*New session started. Send a message to see pipeline details.*"


def create_demo():
    """Create and return the Gradio demo interface."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio is required for the demo. Install with: pip install gradio")

    with gr.Blocks(
        title="Psychologist Agent - Pipeline Demo"
    ) as demo:
        gr.Markdown("""
        # Psychologist Agent Demo

        **Knowledge Distillation + DPO Fine-tuning + Multi-Layer Safety**

        A privacy-preserving mental health AI: cloud model (Deepseek-V3) analyzes with redacted text,
        local model (Llama-3.1-8B GGUF) generates responses with original text.

        > **Disclaimer**: This is a demonstration system for educational purposes.
        > Not a substitute for professional mental health care.
        > **Crisis**: Call 988 | Text HOME to 741741 | Emergency: 911

        ---
        """)

        with gr.Row():
            # Left: Chat interface
            with gr.Column(scale=3):
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

            # Right: Pipeline Details Panel
            with gr.Column(scale=2):
                gr.Markdown("## Pipeline Details")
                pipeline_output = gr.Markdown(
                    value="*Send a message to see pipeline details.*",
                    label="Pipeline Internals"
                )

        with gr.Accordion("Suggested Demo Scenarios", open=False):
            gr.Markdown("""
            ### Scenario 1: Normal Conversation (Anxiety)
            > `I've been feeling really anxious about my upcoming exams`

            ### Scenario 2: Crisis Intervention
            > `I want to end my life`

            ### Scenario 3: PII Protection
            > `My name is John Smith, email john@example.com, I feel depressed`

            ### Scenario 4: Multi-Turn Memory
            Send 3-4 messages in sequence to see UserProfile accumulate.

            ---
            **Current Mode**: `{mode}` | Set `LLM_TYPE=LOCAL` for real GGUF inference
            """.format(mode=os.getenv("LLM_TYPE", "MOCK")))

        with gr.Accordion("Architecture Overview", open=False):
            gr.Markdown("""
            ### Two-Pipeline Architecture

            **Pipeline 1: Offline Training (Knowledge Distillation)**
            ```
            Counsel Chat (2,775) → Clean (863) → Augment (1,057)
                → Baseline Gen → Deepseek Judge (704 passed, 66.6%)
                → DPO Train (633) / Eval (71) → QLoRA → GGUF (4.6GB)
            ```

            **Pipeline 2: Online Inference**
            ```
            User Input → [1] Safety Gateway (BGE-small, 217 patterns)
                       → [2] PII Redaction (Presidio + Regex)
                       → [3] RAG (FAISS, CBT/DBT/WHO, top-5)
                       → [4] Cloud Analysis (Deepseek-V3, 10-turn history)
                       → [5] Risk Audit (keyword + cloud dual-check)
                       → [6] Local Generation (GGUF, 3-turn history)
                       → Response
            ```

            ### Key Design Decisions
            - **Two-Prompt System**: Cloud sees redacted text + long history;
              Local sees original text + short history + supervisor guidance
            - **Safety: Only-Escalate**: Risk can only go up, never down
            - **Privacy**: PII stripped before cloud; UserProfile = structured summary only
            """)

        # Event handlers
        submit.click(
            run_chat,
            inputs=[msg, chatbot],
            outputs=[chatbot, pipeline_output]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        msg.submit(
            run_chat,
            inputs=[msg, chatbot],
            outputs=[chatbot, pipeline_output]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        clear.click(
            clear_chat,
            outputs=[chatbot, pipeline_output]
        )

        new_session.click(
            clear_chat,
            outputs=[chatbot, pipeline_output]
        )

    return demo


def main():
    """Run the demo."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    main()
