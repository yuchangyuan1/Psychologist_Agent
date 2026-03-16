# Psychologist AI Agent

## 1. Project Overview

**Psychologist AI Agent** is a professional mental health counseling AI system built with **Knowledge Distillation** (Deepseek-V3 -> Llama-3.1-8B) and **Direct Preference Optimization (DPO)**. The system is designed for privacy-preserving local deployment on consumer-grade hardware (8GB VRAM).

### Core Design Philosophy

- **Cloud for analysis, Local for generation**: Deepseek-V3 (cloud) performs clinical analysis on redacted text; Llama-3.1-8B (local GGUF) generates natural responses with original text
- **Multi-layer safety**: Semantic safety gateway + PII redaction + risk audit + crisis intervention
- **Knowledge distillation**: Professional counselor responses as chosen, baseline model responses as rejected, filtered by automated judge

## 2. System Architecture

### Two-Pipeline Design

```
Pipeline 1: Offline Training (Knowledge Distillation)
===========================================================
Counsel Chat (2,775) --> Clean & Dedup (863, 31%)
    --> Deepseek Augmentation (1,057, +194 for low-freq topics)
    --> Baseline Generation (generic prompt, intentionally weak)
    --> Deepseek Judge (704 passed, 66.6% acceptance rate)
    --> DPO Dataset (633 train / 71 eval)
    --> QLoRA Fine-tuning (Unsloth, Colab T4)
    --> GGUF Export (Q4_K_M, 4.6 GB)

Pipeline 2: Online Inference (Real-time Interaction)
===========================================================
User Input
    --> [1] Safety Gateway (BGE-small embeddings, 217 risk patterns)
    --> [2] PII Redaction (Presidio NER + Regex, 7 entity types)
    --> [3] RAG Retrieval (FAISS, CBT/DBT/WHO, top-5 chunks)
    --> [4] Cloud Analysis (Deepseek-V3 Supervisor, 10-turn history + UserProfile)
            - Input: redacted text only (privacy preserved)
            - Output: JSON {risk_level, guidance, technique, profile_update}
    --> [5] Risk Audit (keyword hard-check + cloud analysis, only-escalate policy)
    --> [6] Local Generation (GGUF, 3-turn history + supervisor guidance)
            - Input: original text (full context)
            - The local model does NOT know about the supervisor
    --> Response to User
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Two-Prompt System** | Cloud prompt (Prompt 1) drives analysis with long context; Local prompt (Prompt 2) guides generation with short context + supervisor guidance |
| **PII Before Cloud** | Sensitive data never leaves local machine; cloud sees only redacted text |
| **Only-Escalate Safety** | Risk level can only go up (Safety Gateway -> Risk Audit), never down |
| **Weak Baseline for DPO** | Generic prompt (not therapy-specific) creates clear chosen/rejected gap |
| **Judge Filtering** | 66.6% pass rate proves genuine quality filtering, not rubber-stamping |
| **Factory Pattern** | `LLMFactory` enables MOCK/CLOUD/LOCAL switching via `LLM_TYPE` env var |

## 3. Tech Stack

| Component | Technology |
|-----------|-----------|
| **Base Model** | Llama-3.1-8B-Instruct |
| **Fine-tuning** | DPO + QLoRA (r=16, 4-bit), 3 epochs, beta=0.1 |
| **Training Acceleration** | [Unsloth](https://github.com/unslothai/unsloth) (2-3x speedup, 50% VRAM savings) |
| **Cloud Supervisor** | Deepseek-V3 API |
| **Local Inference** | llama-cpp-python (GGUF, Q4_K_M) |
| **Safety Gateway** | BAAI/bge-small-en-v1.5 semantic matching |
| **PII Redaction** | Presidio Analyzer + custom regex patterns |
| **RAG** | FAISS + sentence-transformers |
| **Knowledge Base** | CBT techniques, DBT skills, WHO suicide prevention guide |
| **Frontend** | Gradio |
| **Testing** | pytest + pytest-asyncio (215 tests) |
| **Language** | Python 3.10+ (compatible with 3.13) |

## 4. Directory Structure

```
psychologist-agent/
├── configs/                    # Configuration files
├── data/
│   ├── processed/              # Cleaned data (863 + 1,057 augmented)
│   ├── dpo/                    # DPO training triplets (633 train / 71 eval)
│   ├── knowledge/              # RAG knowledge base (CBT, DBT, WHO)
│   ├── vectordb/               # Pre-built FAISS index
│   ├── safety/                 # 217 risk patterns + safety responses
│   └── crisis/                 # Crisis intervention templates & resources
├── models/                     # GGUF model + QLoRA weights
├── src/
│   ├── api/                    # Deepseek client + data models
│   ├── audit/                  # Risk checker + crisis handler + audit logger
│   ├── inference/              # GGUF local inference (llama-cpp-python)
│   ├── memory/                 # Conversation memory + user profiles
│   ├── privacy/                # PII detection & redaction
│   ├── prompt/                 # Two-prompt generator (cloud + local)
│   ├── rag/                    # FAISS retriever + document loader
│   ├── safety/                 # Semantic safety gateway (BGE embeddings)
│   ├── session/                # Session lifecycle management
│   ├── utils/                  # LLM factory + logging
│   └── main.py                 # Main orchestrator (complete pipeline)
├── scripts/                    # Data processing & evaluation scripts
├── notebooks/                  # Colab notebooks (Unsloth)
├── tests/                      # 10 test files, 215 tests
├── demo/                       # Gradio web demo
└── Evaluation                  # Model evaluation
```

## 5. Installation & Setup

### 5.1 Requirements
- Python 3.10+
- CUDA Toolkit 12.x (for GPU acceleration)
- **Hardware**: 8GB+ VRAM GPU (tested on RTX 4060 Laptop)

### 5.2 Install Dependencies

```bash
git clone https://github.com/yuchangyuan1/Psychologist_Agent.git
cd Psychologist_Agent

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

**GPU-accelerated llama-cpp-python** (Python 3.13, compile from source):
```bash
# Windows: Run vcvars64.bat first, then:
set CMAKE_ARGS=-DGGML_CUDA=on
set CMAKE_GENERATOR=Ninja
set FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir
```

**Download GGUF model**:
- [meta-llama-3.1-8b-instruct.Q4_K_M.gguf](https://drive.google.com/file/d/1PAG7wG-PBuw8FsXIfQ8OPSvZrLxAqhWI/view?usp=drive_link) (4.6 GB) -> place in `models/Psychologist_Agent`

### 5.3 Configuration

Create a `.env` file:
```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_TYPE=LOCAL    # MOCK (demo/testing) | CLOUD | LOCAL
```

## 6. Usage

### 6.1 Run Demo (Gradio Web UI)

**MOCK mode** (no model needed, for testing):
```cmd
cd /d "D:\yuchangyuan\Documents\Psychologist_Agent"
python -m demo.app
```

**LOCAL mode** (uses local GGUF model, requires llama-cpp-python + GPU):
```cmd
cd /d "D:\yuchangyuan\Documents\Psychologist_Agent"
set LLM_TYPE=LOCAL
set LOCAL_MODEL_PATH=models/meta-llama-3.1-8b-instruct.Q4_K_M.gguf
set DEEPSEEK_API_KEY=sk-xxxxxxxx
python -m demo.app
```

> **Note:** Use `cmd.exe`, not PowerShell. In PowerShell `set` does not set environment variables — use `$env:LLM_TYPE = "LOCAL"` instead.

Opens at http://localhost:7860. The demo includes a **Pipeline Details** panel showing each stage's output in real-time.

### 6.2 Run CLI
```bash
python -m src.main
```

### 6.3 Data Processing & Training Pipeline
```bash
# Pipeline 1 steps (in order):
python scripts/data_preparation.py          # Clean & deduplicate
# Run notebooks/generate_baseline_unsloth.ipynb on Colab (Deepseek augmentation + baseline generation)
python scripts/deepseek_judge.py             # Quality judgment & filtering
# Run notebooks/train_dpo.ipynb on Colab (DPO fine-tuning + GGUF export)

# Supporting scripts:
python scripts/process_who_pdf.py            # Process WHO PDF
```

### 6.4 Run Tests
```bash
pytest tests/ --cov=src       # All 215 tests with coverage
pytest tests/test_e2e_pipeline.py  # E2E pipeline tests only
```
> All tests use `LLM_TYPE=MOCK` - no real API calls or model loading required.

## 7. Results

### Pipeline 1: Data & Training

| Stage | Count | Notes |
|-------|-------|-------|
| Raw data (Counsel Chat) | 2,775 | Original dataset |
| Cleaned & deduplicated | 863 (31%) | Strict deduplication |
| Deepseek augmented | 1,057 (+194) | 18 low-frequency topics augmented to >= 20 |
| Baseline generated | 1,057 pairs | Generic prompt (intentionally weak) |
| Judge passed | 704 (66.6%) | 5-dimension scoring, temperature=0.1 |
| DPO training set | 633 | High-quality preference pairs |
| DPO evaluation set | 71 | Held-out evaluation |

### DPO Evaluation Results

| Metric | Score |
|--------|-------|
| DPO vs Baseline win rate | 90% (9/10 samples) |
| Empathy improvement | +1.70 |
| Safety improvement | +1.00 |

### Pipeline 2: Inference Performance

| Metric | Result |
|--------|--------|
| Local inference speed | ~43.5 tokens/s (RTX 4060 Laptop, 8GB VRAM) |
| GGUF model size | 4.6 GB (Q4_K_M quantization) |
| VRAM usage | ~5-6 GB (full GPU offload) |
| Safety risk patterns | 217 patterns (English + Chinese) |
| RAG knowledge chunks | 179 chunks (FAISS index, 269KB) |
| Test coverage | 215 tests across 10 test files |

## 8. Module Overview

| Module | Purpose | Key Features |
|--------|---------|-------------|
| `src/safety/` | Semantic safety gateway | BGE-small embeddings, 3-tier thresholds (0.75/0.80/0.85), 217 patterns |
| `src/privacy/` | PII redaction | Presidio NER + regex, 7 entity types, reversible masking |
| `src/rag/` | Knowledge retrieval | FAISS vector search, CBT/DBT/WHO docs, top-5 retrieval |
| `src/api/` | Cloud API client | Deepseek-V3 async client, JSON parsing, mock mode |
| `src/prompt/` | Two-prompt system | Cloud supervisor prompt + local agent prompt |
| `src/audit/` | Risk assessment | Keyword hard-check + cloud analysis, only-escalate policy |
| `src/inference/` | Local GGUF inference | llama-cpp-python, streaming, full GPU offload |
| `src/memory/` | Conversation memory | Cloud 10-turn + Local 3-turn history, UserProfile (living document) |
| `src/session/` | Session management | Timeout handling, concurrent sessions |
| `src/main.py` | Main orchestrator | Complete 8-step pipeline with graceful degradation |
## 9. Evaluation & Benchmarks

To validate the reliability, clinical accuracy, and operational efficiency of the Psychologist AI Agent, we conducted a rigorous four-stage evaluation. Detailed execution logs can be reproduced using `Benchmark1.ipynb` and `Benchmark2.ipynb`.

### 1. Safety Gateway Benchmarking
**Objective:** Evaluate the local semantic gateway's resilience against malicious triggers (self-harm, medical non-compliance, violence) prior to any reasoning execution.

**Methodology:** Tested against a curated dataset of 100 prompts (70 risky / 30 safe) covering self-harm, violence, and medical advice categories. The gateway uses a two-stage approach: fast keyword hard-check (O(n) string matching) followed by BGE-Small semantic similarity against 217 risk patterns.

* **Average Latency:** **11.26 ms** | **P95 Latency:** 20.74 ms
* **Design Note:** The gateway is intentionally tuned for **zero missed detections** (Recall 1.00 on risky content) — in a mental health context, failing to catch a crisis signal is far more costly than a false positive. Legitimate safe inputs that are flagged are handled gracefully by downstream pipeline stages.

| Metric (Local Semantic Engine Only) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Safe/Clean** | 1.00 | 0.20 | 0.33 |
| **Risky/Flagged** | 0.74 | 1.00 | 0.85 |

### 2. RAG Pipeline Precision
**Objective:** Measure the retrieval accuracy of the FAISS-based knowledge system to ensure responses are grounded in verified clinical frameworks (CBT/DBT/WHO), thereby suppressing medical hallucinations.

**Methodology:** Evaluated 4 representative clinical queries (CBT techniques, suicidal crisis, DBT distress tolerance, safety planning) against an index of 179 professional therapeutic chunks.

* **Top-5 Hit Rate:** **100%** (4/4 representative queries)
* **Mean Reciprocal Rank (MRR):** **0.875**
* **Average Retrieval Latency:** **19.0 ms**

### 3. Agent Quality Assessment
**Objective:** Assess the therapeutic response quality using an independent AI judge.

**Methodology:** Utilized GPT-4o-mini as an independent judge to grade (scale 1–10) 15 randomly sampled agent responses from the 100-prompt benchmark. The benchmark dataset consists of 70 risky prompts (self-harm, violence, medical) and 30 safe prompts; many responses are crisis interventions rather than standard counseling exchanges, which affects the empathy/accuracy scores compared to a pure counseling context.

| Psychological Dimension | DPO Agent Score |
| :--- | :---: |
| **Empathy** | **5.33** |
| **Clinical Accuracy** | **4.60** |
| **Safety Compliance** | **8.47** |

### 4. Inference Efficiency
**Objective:** Verify the feasibility of deploying the agent entirely on standard, consumer-grade hardware without compromising the 8-step pipeline execution.

**Methodology:** End-to-end load testing on 100 prompts on Colab A100 GPU.

* **Target Hardware:** Consumer-grade GPUs (e.g., NVIDIA RTX 3060 / Colab Tesla T4)
* **Local inference speed:** ~43.5 tokens/s (GGUF generation only, RTX 4060 Laptop)
* **End-to-end pipeline latency:** ~6.0 s/request average on A100 (includes Safety + RAG + DeepSeek cloud API + local generation; safety-intercepted prompts complete faster)
* **VRAM Consumption:** **< 6.5 GB** peak usage during local text generation.
* **Model Format:** 4-bit GGUF quantization (`Q4_K_M`) via `llama-cpp-python` with full GPU offloading.
* **Privacy:** Presidio + regex PII redaction pipeline applied before all cloud API calls; sensitive data is processed locally only.
  
## 10. Disclaimer

This system is an AI assistant for educational and research purposes only.
- Not a substitute for professional mental health care or licensed therapy.
- If you or someone you know is in crisis, please call **988** (Suicide & Crisis Lifeline) or your local emergency number.
- The developers assume no liability for consequences arising from use of this system.
