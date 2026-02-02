# Psychologist AI Agent

## 1. 项目简介 (Project Overview)

**Psychologist AI Agent** 是一个专业的心理咨询 AI 代理系统。该项目旨在通过知识蒸馏 (Knowledge Distillation) 和直接偏好优化 (DPO - Direct Preference Optimization) 技术，构建一个安全、专业且高效的本地化心理咨询对话系统。

本项目采用 **混合云端工作流 (Hybrid Cloud-Local Workflow)**，结合了云端大模型（Deepseek-V3）的强大推理能力与本地模型（Llama-3.1-8B-Instruct 微调版）的隐私保护优势，特别针对消费级硬件（如 8GB 显存 GPU）进行了优化。

### 核心目标
*   提供共情、专业的心理咨询对话。
*   保障用户隐私（本地推理 + PII 脱敏）。
*   确保交互安全（语义安全网关 + 危机干预机制）。
*   实现低延迟、低成本的本地部署。

## 2. 系统架构 (System Architecture)

项目分为两个核心 Pipeline：

1.  **Pipeline 1: Offline Training (离线训练)**
    *   数据清洗与准备 (Counsel Chat Dataset)。
    *   偏好数据集构建 (使用 Deepseek-V3 作为 Judge)。
    *   DPO 微调与 QLoRA 适配 (**Unsloth 加速**)。
    *   模型量化 (GGUF 4-bit, Unsloth 内置导出)。

2.  **Pipeline 2: Online Inference (在线推理)**
    *   **安全网关**: 实时检测高风险输入。
    *   **RAG 系统**: 检索 CBT/DBT 等专业心理治疗知识。
    *   **隐私保护**: PII (个人身份信息) 自动脱敏。
    *   **混合推理**: 云端 Supervisor (Deepseek) 分析策略 -> 本地 Agent (Llama) 生成回复。
    *   **危机干预**: 针对高危情况（如自杀倾向）触发紧急响应流程。

## 3. 技术栈 (Tech Stack)

| 组件 | 技术选型 |
|------|----------|
| **基础模型** | Llama-3.1-8B-Instruct |
| **微调方法** | DPO + QLoRA (4-bit quantization) |
| **训练加速** | [Unsloth](https://github.com/unslothai/unsloth) (2-3x 加速, 50% 显存节省) |
| **云端推理** | Deepseek-V3 API |
| **向量数据库** | FAISS / ChromaDB |
| **Embedding** | BAAI/bge-small-en-v1.5 |
| **RAG 框架** | LangChain |
| **训练框架** | Unsloth + trl (DPOTrainer) |
| **推理框架** | llama.cpp (GGUF) + llama-cpp-python |
| **开发语言** | Python 3.10+ |

## 4. 目录结构 (Directory Structure)

```
psychologist-agent/
├── configs/                    # 配置文件 (API, DPO, Inference)
├── data/                       # 数据资产
│   ├── raw/                    # 原始数据 (PDF, JSON)
│   ├── processed/              # 清洗后数据
│   ├── dpo/                    # DPO 训练三元组
│   ├── knowledge/              # RAG 知识库 (Markdown)
│   ├── vectordb/               # 向量索引
│   ├── safety/                 # 风险模式库
│   └── crisis/                 # 危机干预资源
├── models/                     # 模型权重 (QLoRA, GGUF)
├── prompts/                    # Prompt 模板
├── src/                        # 源代码
│   ├── api/                    # Deepseek 客户端
│   ├── audit/                  # 风险审计与危机处理
│   ├── inference/              # 本地推理服务
│   ├── memory/                 # 会话记忆
│   ├── privacy/                # PII 脱敏
│   ├── prompt/                 # Prompt 生成器
│   ├── rag/                    # RAG 检索器
│   ├── safety/                 # 安全网关
│   ├── session/                # 会话管理
│   ├── utils/                  # 工具函数 (LLM Factory)
│   └── main.py                 # 主入口
├── scripts/                    # 数据处理与评判脚本
├── notebooks/                  # Colab Notebooks (Unsloth)
│   ├── generate_baseline_unsloth.ipynb  # Baseline 生成
│   └── train_dpo.ipynb                  # DPO 训练
├── tests/                      # 单元与集成测试
├── demo/                       # 演示应用 (Gradio/Streamlit)
├── logs/                       # 运行日志
└── plan.md                     # 详细项目计划书
```

## 5. 安装与配置 (Installation & Setup)

### 5.1 环境要求
*   Python 3.10 或更高版本
*   CUDA Toolkit 12.x (推荐用于 GPU 加速)
*   **硬件**: 至少 8GB RAM (推荐 16GB+), GPU 显存 6GB+ (推荐 8GB+)

### 5.2 安装依赖

1.  克隆仓库：
    ```bash
    git clone https://github.com/yuchangyuan1/6895_project_Agent
    cd 6895_project_Agent
    ```

2.  创建并激活虚拟环境：
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  安装 Python 依赖：
    ```bash
    pip install -r requirements.txt
    ```

4.  安装 `llama-cpp-python` (GPU 版)：
    为了启用 GPU 加速，请按照您的 CUDA 版本安装对应的预编译包。例如 (CUDA 12.1)：
    ```bash
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
    ```

### 5.3 配置文件

在项目根目录创建 `.env` 文件（参考 `.env.example`，如果有）：

```env
# Deepseek API 配置
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# LLM 类型配置 (MOCK / CLOUD / LOCAL)
LLM_TYPE=LOCAL
```

## 6. 使用指南 (Usage)

### 6.1 运行 Demo 应用
启动基于 Web 的交互演示：
```bash
python demo/app.py
```

### 6.2 运行主程序 (CLI)
```bash
python src/main.py
```

### 6.3 数据处理与训练 (开发人员)
*   **数据准备**: `python scripts/data_preparation.py`
*   **Baseline 生成**: 在 Colab 运行 `notebooks/generate_baseline_unsloth.ipynb` (Unsloth)
*   **Judge 评判**: `python scripts/deepseek_judge.py`
*   **DPO 训练**: 在 Colab 运行 `notebooks/train_dpo.ipynb` (Unsloth)

## 7. 免责声明 (Disclaimer)

**重要提示**：
本系统是一个人工智能助手，**不是**人类心理医生或执业医师。
*   它提供的建议仅供参考，**不能**替代专业的医疗诊断或治疗。
*   如果您或您认识的人正处于紧急危机中（如自杀、自伤或伤害他人），请**立即**拨打当地急救电话或联系专业危机干预机构（如 988 或 110/120）。
*   开发者不对使用本系统产生的任何后果负责。


