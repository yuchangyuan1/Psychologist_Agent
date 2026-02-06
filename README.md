# Psychologist AI Agent

## 1. Project Overview

**Psychologist AI Agent** is a professional mental health counseling AI agent system. This project aims to build a safe, professional, and efficient localized psychological counseling dialogue system through Knowledge Distillation and Direct Preference Optimization (DPO) techniques.

The project adopts a **Hybrid Cloud-Local Workflow**, combining the powerful reasoning capabilities of a cloud-based large model (Deepseek-V3) with the privacy-preserving advantages of a local model (fine-tuned Llama-3.1-8B-Instruct), specifically optimized for consumer-grade hardware (e.g., GPUs with 8GB VRAM).

### Core Objectives
*   Provide empathetic, professional psychological counseling conversations.
*   Protect user privacy (local inference + PII redaction).
*   Ensure interaction safety (semantic safety gateway + crisis intervention mechanism).
*   Achieve low-latency, low-cost local deployment.

## 2. System Architecture

The project consists of two core pipelines:

1.  **Pipeline 1: Offline Training**
    *   Data cleaning and preparation (Counsel Chat Dataset).
    *   Preference dataset construction (using Deepseek-V3 as Judge).
    *   DPO fine-tuning with QLoRA adaptation (**Unsloth accelerated**).
    *   Model quantization (GGUF 4-bit, Unsloth built-in export).

2.  **Pipeline 2: Online Inference**
    *   **Safety Gateway**: Real-time detection of high-risk inputs.
    *   **RAG System**: Retrieval of professional therapeutic knowledge (CBT/DBT).
    *   **Privacy Protection**: Automatic PII (Personally Identifiable Information) redaction.
    *   **Hybrid Inference**: Cloud Supervisor (Deepseek) analyzes strategy -> Local Agent (Llama) generates response.
    *   **Crisis Intervention**: Emergency response triggered for high-risk situations (e.g., suicidal ideation).

## 3. Tech Stack

| Component | Technology |
|-----------|-----------|
| **Base Model** | Llama-3.1-8B-Instruct |
| **Fine-tuning Method** | DPO + QLoRA (4-bit quantization) |
| **Training Acceleration** | [Unsloth](https://github.com/unslothai/unsloth) (2-3x speedup, 50% VRAM savings) |
| **Cloud Inference** | Deepseek-V3 API |
| **Vector Database** | FAISS / ChromaDB |
| **Embedding** | BAAI/bge-small-en-v1.5 |
| **RAG Framework** | LangChain |
| **Training Framework** | Unsloth + trl (DPOTrainer) |
| **Inference Framework** | llama.cpp (GGUF) + llama-cpp-python |
| **Language** | Python 3.10+ (compatible with 3.13) |

## 4. Directory Structure

```
psychologist-agent/
├── configs/                    # Configuration files (API, DPO, Inference)
├── data/                       # Data assets
│   ├── raw/                    # Raw data (PDF, JSON)
│   ├── processed/              # Cleaned data
│   ├── dpo/                    # DPO training triplets
│   ├── knowledge/              # RAG knowledge base (Markdown)
│   ├── vectordb/               # Vector index
│   ├── safety/                 # Risk pattern database
│   └── crisis/                 # Crisis intervention resources
├── models/                     # Model weights (QLoRA, GGUF)
├── prompts/                    # Prompt templates
├── src/                        # Source code
│   ├── api/                    # Deepseek client
│   ├── audit/                  # Risk audit & crisis handling
│   ├── inference/              # Local inference service
│   ├── memory/                 # Conversation memory
│   ├── privacy/                # PII redaction
│   ├── prompt/                 # Prompt generator
│   ├── rag/                    # RAG retriever
│   ├── safety/                 # Safety gateway
│   ├── session/                # Session management
│   ├── utils/                  # Utility functions (LLM Factory)
│   └── main.py                 # Main entry point
├── scripts/                    # Data processing & evaluation scripts
├── notebooks/                  # Colab Notebooks (Unsloth)
│   ├── generate_baseline_unsloth.ipynb  # Baseline generation
│   └── train_dpo.ipynb                  # DPO training
├── tests/                      # Unit & integration tests
├── demo/                       # Demo application (Gradio)
└── logs/                       # Runtime logs
```

## 5. Installation & Setup

### 5.1 Requirements
*   Python 3.10 or higher
*   CUDA Toolkit 12.x (recommended for GPU acceleration)
*   **Hardware**: At least 8GB RAM (16GB+ recommended), GPU VRAM 6GB+ (8GB+ recommended)

### 5.2 Install Dependencies

1.  Clone the repository:
    ```bash
    git clone https://github.com/yuchangyuan1/Psychologist_Agent.git
    cd Psychologist_Agent
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Install `llama-cpp-python` (GPU version):
    Python 3.13 has no prebuilt CUDA wheels and requires compilation from source. Ensure CUDA Toolkit 12.x and VS 2022 Build Tools are installed:
    ```bash
    # Windows: Run vcvars64.bat first to configure the build environment, then:
    set CMAKE_ARGS=-DGGML_CUDA=on
    set CMAKE_GENERATOR=Ninja
    set FORCE_CMAKE=1
    pip install llama-cpp-python --no-cache-dir
    ```

5.  Download the GGUF model file:
    The quantized model file is hosted on Google Drive. Download it and place it in the `models/` directory:
    - [meta-llama-3.1-8b-instruct.Q4_K_M.gguf](https://drive.google.com/file/d/1PAG7wG-PBuw8FsXIfQ8OPSvZrLxAqhWI/view?usp=drive_link) (4.6 GB)

### 5.3 Configuration

Create a `.env` file in the project root directory:

```env
# Deepseek API configuration
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# LLM type configuration (MOCK / CLOUD / LOCAL)
LLM_TYPE=LOCAL
```

## 6. Usage

### 6.1 Run Demo Application
Launch the web-based interactive demo:
```bash
python demo/app.py
```

### 6.2 Run Main Program (CLI)
```bash
python src/main.py
```

### 6.3 Data Processing & Training
*   **Data Preparation**: `python scripts/data_preparation.py`
*   **Data Augmentation**: `python scripts/data_augmentation.py` (Deepseek-V3 synthetic augmentation)
*   **Baseline Generation**: Run `notebooks/generate_baseline_unsloth.ipynb` on Colab (Unsloth)
*   **Judge Evaluation**: `python scripts/deepseek_judge.py`
*   **DPO Training**: Run `notebooks/train_dpo.ipynb` on Colab (Unsloth)
*   **Vector Database Build**: `python scripts/build_vectordb.py`
*   **Model Evaluation**: `python scripts/evaluate_model_quality.py`
*   **GGUF Inference Test**: `python scripts/test_gguf_inference.py`
*   **Local Test**: `python scripts/run_local_test.py`
*   **WHO PDF Processing**: `python scripts/process_who_pdf.py`

### 6.4 Run Tests
```bash
# Run all tests
pytest tests/ --cov=src

# Run a single test file
pytest tests/test_safety_gateway.py
```
> Note: All tests use `LLM_TYPE=MOCK` mode and do not require loading real models or calling external APIs.

## 7. Project Results

### Pipeline 1: Offline Training

| Metric | Result |
|--------|--------|
| Raw data (Counsel Chat) | 2,608 samples |
| Cleaned data (deduplicated) | 863 samples |
| Augmented data (Deepseek-V3 synthetic) | 1,057 samples |
| Judge evaluation passed | 704 samples (66.6%) |
| DPO training set | 633 samples |
| DPO validation set | 71 samples |
| Final model | `meta-llama-3.1-8b-instruct.Q4_K_M.gguf` (4.6 GB) |

### Pipeline 2: Online Inference

| Metric | Result |
|--------|--------|
| Local inference speed | ~43.5 tokens/s (RTX 4060 Laptop, full GPU offload) |
| GGUF VRAM usage | ~5-6 GB (Q4_K_M) |
| Safety risk pattern database | 217 patterns |
| RAG knowledge base | CBT/DBT + WHO Suicide Prevention documents |
| Vector index | FAISS (`faiss_index.index`, 269KB) |

## 8. Disclaimer

This system is an AI assistant, not a human psychologist or licensed physician.
*   The advice it provides is for reference only and cannot replace professional medical diagnosis or treatment.
*   If you or someone you know is in an emergency crisis (e.g., suicide, self-harm, or harm to others), please immediately call your local emergency number or contact a professional crisis intervention service.
*   The developers are not responsible for any consequences arising from the use of this system.
