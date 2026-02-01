# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Psychologist Agent** - A mental health AI system using knowledge distillation (Deepseek-V3 → Llama-3.1-8B) and DPO fine-tuning. Designed for privacy-preserving local deployment with 8GB VRAM.

## Architecture

### Two-Pipeline Design

**Pipeline 1: Offline Training (Distillation Phase)**
- Data preparation from Counsel Chat dataset
- Baseline response generation + Deepseek-V3 quality judgment
- DPO training with QLoRA (4-bit) on Colab
- GGUF model conversion for local deployment

**Pipeline 2: Online Inference (Real-time Interaction)**
```
User Input → Safety Gateway → PII Redaction → RAG Retrieval
    → Cloud Analysis (Deepseek-V3) → Risk Audit
    → Local Generation (GGUF) → Memory Update → Response
```

### Key Design Decisions

- **Two-Prompt System**: Prompt 1 drives Deepseek analysis; Prompt 2 guides local GGUF generation
- **Multi-Layer Safety**: Pre-API safety gateway + post-API risk audit + crisis intervention
- **Factory Pattern**: `LLMFactory` enables MOCK/CLOUD/LOCAL switching via `LLM_TYPE` env var
- **PII Before Cloud**: Local redaction prevents sensitive data from reaching external APIs

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# GPU support (Windows CUDA 12.1)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**Local Inference**: Use `src.utils.llm_factory`, NOT transformers + bitsandbytes:
```python
from src.utils.llm_factory import LLMFactory
model = LLMFactory.create_llm()  # Reads LLM_TYPE from env
```

## Testing

```bash
pytest tests/ --cov=src           # All tests with coverage
pytest tests/test_module_name.py  # Single test file
```

**Critical Rules**:
- **FORBIDDEN**: Real Deepseek API calls or loading GGUF models in unit tests
- **REQUIRED**: Use `LLM_TYPE=MOCK` or `unittest.mock` for all tests

## Environment Variables

| Variable | Values | Purpose |
|----------|--------|---------|
| `LLM_TYPE` | `MOCK`, `CLOUD`, `LOCAL` | LLM backend selection |
| `LOCAL_MODEL_PATH` | Path to `.gguf` file | Local model location |
| `DEEPSEEK_API_KEY` | `sk-...` | Cloud API credentials |
| `DEEPSEEK_BASE_URL` | API endpoint URL | Cloud API endpoint |

## Module Overview

| Module | Purpose |
|--------|---------|
| `src/utils/` | LLM factory (`llm_factory.py`), logging config |
| `src/api/` | Deepseek-V3 async client |
| `src/safety/` | Semantic safety gateway (BGE-small embeddings) |
| `src/rag/` | Vector retrieval (FAISS/ChromaDB, CBT/DBT knowledge) |
| `src/privacy/` | PII detection/masking (presidio-analyzer) |
| `src/prompt/` | Cloud vs Local prompt generation |
| `src/audit/` | Risk checking, crisis intervention |
| `src/inference/` | GGUF inference via llama-cpp-python |
| `src/memory/` | Conversation history (≥20 turns/session) |
| `src/session/` | User session management |

## Coding Standards

- **Path handling**: MUST use `os.path.join()` or `pathlib` - never hardcode `/` or `\`
- **Async**: Required for all I/O operations (API calls, DB queries)
- **Docstrings**: Google Style

## Language Policy

- **Code & Files**: All code, comments, commit messages, and documentation files must be in **English**
- **User Communication**: Communicate with users in **Chinese (中文)**

## Git Policy

- **Never commit**: `*.env`, `configs/api_config.yaml`, `models/*`, `data/*` (except `.gitkeep`)
- **Workflow**: Feature branch → Local test (MOCK mode) → PR to `main`
- **Repo**: `https://github.com/yuchangyuan1/6895_project_Agent`
