# Psychologist AI Agent 项目计划书

## 1. 项目概述

### 1.1 项目目标
构建一个专业的心理咨询 AI Agent，通过知识蒸馏和 DPO (Direct Preference Optimization) 微调技术，在本地部署一个安全、专业、高效的心理咨询对话系统。

### 1.2 系统架构概览

项目分为两个核心 Pipeline：

| Pipeline | 名称 | 目的 |
|----------|------|------|
| Pipeline 1 | Offline Training (Distillation Phase) | 数据准备、偏好数据集构建、DPO 微调、生成 QLoRA 权重 |
| Pipeline 2 | Online Inference (Real-time Interaction) | 实时用户交互、安全过滤、RAG 检索、风险审计、最终响应生成 |

### 1.3 技术栈总览

| 组件 | 技术选型 |
|------|----------|
| 基础模型 | Llama-3.1-8B-Instruct |
| 微调方法 | DPO + QLoRA (4-bit quantization) |
| 云端推理 | Deepseek-V3 API |
| 向量数据库 | FAISS / ChromaDB |
| Embedding 模型 | BAAI/bge-small-en-v1.5 |
| RAG 框架 | LangChain |
| 训练框架 | transformers + trl + peft |
| 推理框架 | llama.cpp (GGUF) + llama-cpp-python |

### 1.4 执行策略：混合云端工作流 (Hybrid Cloud-Local Workflow)

本项目采用混合云端工作流，充分利用本地开发环境和云端计算资源，特别针对 **8GB 显存** 环境进行优化：

1. **本地开发与测试**
   - 在本地电脑编写代码文件并进行单元测试
   - 使用 `llm_factory.py` 配置为 **Mock 模式** 或 **DeepSeek API 模式**，无需加载本地大模型
   - 验证数据流转和业务逻辑

2. **代码版本管理**
   - 通过 `gh` 命令将代码 commit 到 GitHub 仓库
   - 排除大文件和敏感配置

3. **云端训练 (Google Colab)**
   - 利用 Colab T4 (16GB) 资源进行 DPO 微调
   - 训练完成后：
     1. 将 QLoRA 权重合并回 Base Model
     2. 使用 `llama.cpp` 工具将模型转换为 **GGUF 格式**
     3. 进行 4-bit 量化 (Q4_K_M) 以适应本地 8GB 显存
   - 将生成的 `.gguf` 文件下载回本地 `models/` 目录

4. **本地推理**
   - 本地加载 GGUF 模型，享受低延迟、高隐私的推理服务

---

## 2. Pipeline 1: Offline Training (Distillation Phase)

### 2.1 阶段 1.1: 数据准备

#### 2.1.1 目标
准备高质量的心理咨询对话数据集，为后续 DPO 训练提供原始数据。

#### 2.1.2 实施步骤

1. **获取 Counsel Chat Dataset**
   - 数据来源：Hugging Face `nbertagnolli/counsel-chat`
   - 数据格式：心理咨询 Q&A 记录
   - 预期数据量：约 3,000+ 条对话记录

2. **数据清洗与预处理**
   - 去除重复、无效、过短的记录
   - 统一文本格式（去除特殊字符、规范化标点）
   - 过滤敏感/不当内容
   - 分割训练集/验证集/测试集 (80/10/10)

3. **数据结构化**
   ```json
   {
     "id": "unique_id",
     "question": "用户的心理咨询问题",
     "answer": "专业咨询师的回答",
     "topic": "咨询主题分类"
   }
   ```

#### 2.1.3 交付物
- [ ] 清洗后的数据集文件：`data/processed/counsel_chat_cleaned.jsonl`
- [ ] 数据统计报告：`reports/data_statistics.md`
- [ ] 数据质量检查脚本：`scripts/data_validation.py`

#### 2.1.4 验收标准
- 数据集记录数 ≥ 2,500 条
- 无重复记录（去重后）
- 每条记录包含完整的 question 和 answer 字段
- 通过数据质量检查脚本验证

---

### 2.2 阶段 1.2: 偏好数据集构建

#### 2.2.1 目标
构建 DPO 训练所需的 `{prompt, chosen, rejected}` 三元组数据集。

#### 2.2.2 实施步骤

1. **生成 Rejected Responses (Baseline)**
   - 使用基础 Llama 模型对每个 prompt 生成回复
   - 配置推理参数：
     ```python
     generation_config = {
         "max_new_tokens": 512,
         "temperature": 0.7,
         "top_p": 0.9,
         "do_sample": True
     }
     ```
   - 批量推理，保存生成结果

2. **提取 Chosen Responses**
   - 从原始数据集中提取专业咨询师的真实回答
   - 作为 "chosen" (优选) 响应

3. **构建 JSONL 三元组**
   ```json
   {
     "prompt": 用户问题,
     "chosen": 专业咨询师的真实回答,
     "rejected": 基础 Llama 的生成回答
   }
   ```

4. **Deepseek-V3 Quality Judgment**
   - Use Deepseek-V3 API as Judge
   - Evaluation criteria:
     - Professionalism
     - Empathy
     - Safety
     - Helpfulness
   - Judge Prompt Template:
     ```
     As a mental health counseling expert, please evaluate the quality of the following two responses.

     User Question: {prompt}

     Response A: {chosen}
     Response B: {rejected}

     Please determine whether Response A is better than Response B in terms of professionalism, empathy, and safety.
     Output format: {"is_chosen_better": true/false, "reason": "brief explanation"}
     ```

5. **数据筛选与保存**
   - 仅保留 `is_chosen_better == true` 的记录
   - 保存到 DPO 训练数据集

#### 2.2.3 交付物
- [ ] Baseline 生成脚本：`scripts/generate_baseline_responses.py`
- [ ] Judge 评判脚本：`scripts/deepseek_judge.py`
- [ ] DPO 数据集：`data/dpo/train.jsonl`, `data/dpo/eval.jsonl`
- [ ] 评判结果日志：`logs/judge_results.jsonl`

#### 2.2.4 验收标准
- DPO 数据集记录数 ≥ 2,000 条（通过评判的样本）
- 每条记录包含完整的 prompt, chosen, rejected 字段
- Judge 评判通过率统计 ≥ 70%
- 随机抽检 50 条记录，人工验证准确率 ≥ 85%

---

### 2.3 阶段 1.3: DPO 微调

#### 2.3.1 目标
使用 DPO 算法和 QLoRA 技术微调 Llama-3.1-8B-Instruct 模型。

#### 2.3.2 实施步骤

1. **环境配置**
   ```bash
   # 核心依赖
   pip install transformers>=4.36.0
   pip install trl>=0.7.0
   pip install peft>=0.7.0
   pip install accelerate>=0.25.0
   pip install bitsandbytes>=0.41.0
   ```

2. **QLoRA 配置 (4-bit Quantization)**
   ```python
   from transformers import BitsAndBytesConfig

   # 4-bit quantization config
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16,
       bnb_4bit_use_double_quant=True
   )

   lora_config = LoraConfig(
       r=16,                      # LoRA rank
       lora_alpha=32,             # LoRA alpha
       lora_dropout=0.05,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
       bias="none",
       task_type="CAUSAL_LM"
   )
   ```

3. **DPO 训练配置**
   ```python
   training_args = DPOConfig(
       output_dir="./outputs/dpo_psychologist",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,
       learning_rate=5e-5,
       beta=0.1,                  # DPO temperature
       warmup_ratio=0.1,
       logging_steps=10,
       save_steps=500,
       eval_steps=500,
       fp16=True,                 # 或 bf16=True
   )
   ```

4. **训练执行**
   - 加载基础模型和 QLoRA 配置
   - 加载 DPO 数据集
   - 启动 DPOTrainer 训练
   - 监控 loss 曲线和评估指标

5. **模型导出**
   - 保存 QLoRA adapter weights
   - 可选：合并 QLoRA 权重到基础模型

#### 2.3.3 硬件要求
| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3090 24GB | RTX 4090 24GB / A100 40GB |
| RAM | 32GB | 64GB |
| 存储 | 100GB SSD | 200GB NVMe SSD |

#### 2.3.4 交付物
- [ ] 训练脚本：`scripts/train_dpo.py`
- [ ] 训练配置文件：`configs/dpo_config.yaml`
- [ ] QLoRA 权重文件：`models/psychologist_qlora/`
- [ ] 训练日志：`logs/training/`
- [ ] 训练曲线报告：`reports/training_curves.png`

#### 2.3.5 验收标准
- 训练 loss 收敛，无明显震荡
- 验证集 reward accuracy ≥ 60%
- QLoRA 权重可成功加载并推理
- 人工评估：在 20 个测试问题上，微调后模型回答质量明显优于基础模型

---

## 3. Pipeline 2: Online Inference (Real-time Interaction)

### 3.1 阶段 2.1: 安全网关 (Semantic Safety Gateway)

#### 3.1.1 目标
构建语义安全过滤层，识别并拦截高风险输入。

#### 3.1.2 实施步骤

1. **高风险关键词/语义库构建**
   - 自杀相关关键词
   - 自伤相关表述
   - 暴力威胁表述
   - 紧急危机信号

2. **Embedding Model Deployment**
   - Model: `BAAI/bge-small-en-v1.5`
   - Local deployment, no API calls required

3. **语义相似度检测**
   ```python
   def check_safety(user_input: str, threshold: float = 0.85) -> dict:
       """
       返回: {"is_safe": bool, "risk_level": str, "matched_pattern": str}
       """
       input_embedding = model.encode(user_input)
       for risk_pattern in risk_patterns:
           similarity = cosine_similarity(input_embedding, risk_pattern.embedding)
           if similarity > threshold:
               return {
                   "is_safe": False,
                   "risk_level": "high",
                   "matched_pattern": risk_pattern.category
               }
       return {"is_safe": True, "risk_level": "low", "matched_pattern": None}
   ```

4. **Safety Response Templates**
   - Prepare professional safety guidance responses for different risk types
   - Include crisis hotlines and professional help resources

#### 3.1.3 交付物
- [ ] 安全网关模块：`src/safety/gateway.py`
- [ ] 风险模式库：`data/safety/risk_patterns.json`
- [ ] 安全响应模板：`data/safety/safety_responses.json`
- [ ] 单元测试：`tests/test_safety_gateway.py`

#### 3.1.4 验收标准
- 高风险关键词检出率 ≥ 95%
- 误报率 ≤ 5%
- 单次检测延迟 ≤ 50ms
- 覆盖至少 100 个风险模式

---

### 3.2 阶段 2.2: RAG 系统构建

#### 3.2.1 目标
构建本地 RAG 系统，提供 CBT/DBT 等心理治疗知识检索能力。

#### 3.2.2 实施步骤

1. **知识库准备**
   - CBT (认知行为疗法) 相关资料
   - DBT (辩证行为疗法) 相关资料
   - 心理咨询技术指南
   - 常见心理问题应对策略
   - **新增权威资料**: `data/raw/WHO_Preventing_suicide.pdf` (需清洗与分块)

2. **文档处理与分块**
   - **PDF 清洗**: 使用 PyMuPDF 提取 `WHO_Preventing_suicide.pdf` 文本，去除页眉页脚，保留章节结构。
   - **分块策略**:
   ```python
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=600,
       chunk_overlap=100,
       separators=["\n\n", "\n", "。", "！", "？"]
   )
   ```

3. **向量数据库构建**
   - 使用 FAISS 或 ChromaDB
   - Embedding 模型与安全网关共用
   - 支持增量更新

4. **LangChain 检索链配置**
   ```python
   retriever = vectorstore.as_retriever(
       search_type="mmr",           # 最大边际相关性
       search_kwargs={"k": 5, "fetch_k": 10}
   )
   ```

5. **会话记忆存储**
   - 实现短期会话记忆（当前对话上下文）
   - 可选：长期记忆（跨会话用户画像）

#### 3.2.3 交付物
- [ ] 知识库文档：`data/knowledge/`
- [ ] RAG 模块：`src/rag/retriever.py`
- [ ] 向量数据库：`data/vectordb/`
- [ ] 记忆管理模块：`src/memory/store.py`
- [ ] 单元测试：`tests/test_rag.py`

#### 3.2.4 验收标准
- 知识库文档数量 ≥ 50 篇
- 检索相关性（Top-5 命中率）≥ 80%
- 单次检索延迟 ≤ 200ms
- 记忆存储支持至少 20 轮对话历史

---

### 3.3 阶段 2.3: Prompt 生成与 PII 脱敏

#### 3.3.1 目标
构建 Prompt 生成模块，实现 PII (个人身份信息) 脱敏保护。

#### 3.3.2 实施步骤

1. **PII 检测与脱敏**
   - 检测类型：姓名、电话、邮箱、身份证号、地址等
   - 使用正则表达式 + NER 模型
   - 脱敏策略：替换为占位符（如 `[NAME]`, `[PHONE]`）

2. **Prompt Gen 1: Cloud API Prompt (Supervisor)**
   - **输入**: 最近 10 轮对话 + User Profile (JSON) + RAG Context
   - **输出**: 纯 JSON 对象
   ```text
   [System]
   You are a professional mental health counseling supervisor. Your task is to analyze the user's current state, audit for safety risks, and provide strategic guidance for the local counseling agent.

   [Memory - Long Term Profile]
   The following is the structured profile of the user based on previous sessions. Use this to understand long-term context (e.g., diagnosis history, chronic stressors).
   {user_profile_json}

   [Memory - Recent Interaction]
   The following are the raw transcripts of the last 10 conversation turns. Use this to detect immediate emotional shifts and context.
   {recent_history_text}

   [Context - RAG Knowledge]
   Relevant psychological principles or crisis intervention guidelines retrieved for this query:
   {rag_context}

   [User Input]
   {sanitized_user_input}

   [Task]
   1. Analyze the user's risk level (Suicide/Self-harm/Violence).
   2. Identify the primary psychological concern in this specific turn.
   3. Formulate a specific counseling strategy (e.g., "Use Socratic questioning," "Validate emotion").
   4. Update the User Profile if new key facts (e.g., new medication, major life event) are revealed.

   [IMPORTANT]
   Output ONLY a raw JSON object. Do not use Markdown code blocks (```json). Do not add any introductory text. The output must parse directly by Python's json.loads().

   [Output Schema]
   {
     "risk_level": "low" | "medium" | "high",
     "risk_reasoning": "Brief explanation of why this risk level was assigned",
     "primary_concern": "The core issue the user is facing right now",
     "guidance_for_local_model": "Direct instruction to the local agent on how to respond (e.g., 'Acknowledge the anxiety, then ask about their sleep pattern').",
     "suggested_technique": "Specific CBT/DBT technique to apply (e.g., 'Validation', 'Grounding')",
     "updated_user_profile": {
       // Return the updated version of the input [Memory - Long Term Profile]
       // Merge new facts found in this turn into the existing profile. If no change, return the original.
     }
   }
   ```

3. **Prompt Gen 2: Local Model Generation Prompt (Agent)**
   - **机制**: 构建结构化 `messages` 列表供 `create_chat_completion` 使用 (严禁手动拼接 Special Tokens)
   - **System Message Content**:
   ```text
   You are Serenity, a warm, empathetic, and professional mental health counselor. Your goal is to provide a supportive response to the user, following the clinical guidance provided by your supervisor.

   [Supervisor Guidance]
   The following is the clinical analysis of the current situation. You MUST follow this strategy:
   - **Core Issue**: {deepseek_primary_concern}
   - **Strategy to Apply**: {deepseek_suggested_technique}
   - **Instruction**: {deepseek_guidance_for_local_model}

   [Relevant Knowledge]
   {rag_context_snippet}
   ```
   - **User Message Content**:
   ```text
   [Conversation Context]
   (Only the last 3 turns are provided for flow continuity)
   {recent_short_history}

   [User]
   {user_input}

   [Response Guidelines]
   1. Tone: Warm, non-judgmental, and patient.
   2. Safety: If risk is mentioned in Supervisor Guidance, strictly follow crisis protocols.
   3. Style: Keep responses concise (under 150 words) to encourage dialogue. Avoid lecturing.
   4. Privacy: Do not mention that you are receiving instructions from a "supervisor."
   ```

#### 3.3.3 交付物
- [ ] PII 脱敏模块：`src/privacy/pii_redactor.py`
- [ ] Prompt 模板：`prompts/prompt_templates.yaml`
- [ ] Prompt 生成器：`src/prompt/generator.py`
- [ ] 单元测试：`tests/test_pii_redactor.py`

#### 3.3.4 验收标准
- PII 检测召回率 ≥ 95%
- PII 脱敏不影响语义理解
- Prompt 模板可配置、易维护
- 生成的 Prompt 长度控制在 2000 tokens 以内

---

### 3.4 阶段 2.4: 云端 API 集成

#### 3.4.1 目标
安全、稳定地集成 Deepseek-V3 云端 API。

#### 3.4.2 实施步骤

1. **API 客户端封装**
   ```python
   class DeepseekClient:
       def __init__(self, api_key: str, base_url: str):
           self.client = httpx.AsyncClient(...)

       async def analyze(self, prompt: str) -> dict:
           # 带重试、超时、错误处理的 API 调用
           pass
   ```

2. **安全通信**
   - HTTPS 强制
   - API Key 环境变量管理
   - 请求/响应日志脱敏

3. **响应解析与验证**
   - JSON Schema 验证
   - 异常响应处理
   - 降级策略（API 不可用时的本地处理）

4. **审计日志**
   - 记录所有 API 调用（脱敏后）
   - 记录响应时间、状态码
   - 便于问题排查和合规审计

#### 3.4.3 交付物
- [ ] API 客户端：`src/api/deepseek_client.py`
- [ ] 配置管理：`configs/api_config.yaml`
- [ ] 审计日志模块：`src/audit/logger.py`
- [ ] 集成测试：`tests/test_deepseek_integration.py`

#### 3.4.4 验收标准
- API 调用成功率 ≥ 99%（排除网络问题）
- 平均响应时间 ≤ 3s
- 错误重试机制有效
- 审计日志完整且已脱敏

---

### 3.5 阶段 2.5: 风险审计与危机干预

#### 3.5.1 目标
对云端 API 返回的分析结果进行风险审计，高风险情况触发危机干预流程。

#### 3.5.2 实施步骤

1. **风险等级判定**
   - **JSON 清洗**: 使用正则表达式提取第一个有效的 JSON 对象 (e.g., `re.search(r'\{.*\}', text, re.DOTALL)`)
   - 解析 Deepseek-V3 返回的 JSON 对象中的 `risk_level` 字段
   - 必须处理 JSON 解析异常（如果 API 返回非标准 JSON）
   - 结合本地规则进行二次验证

2. **危机干预响应**
   - 当 `risk_level == "high"` 或本地安全网关触发时：
     - 跳过 Llama 生成流程
     - 返回预设的危机干预响应 (Responses stored in `data/crisis/responses.json`)
     - 提供专业求助资源

3. **Crisis Resource Database**
   ```json
   {
     "crisis_hotlines": [
       {"name": "988 Suicide & Crisis Lifeline (US)", "phone": "988"},
       {"name": "Crisis Text Line (US)", "text": "Text HOME to 741741"},
       {"name": "International Association for Suicide Prevention", "url": "https://www.iasp.info/resources/Crisis_Centres/"}
     ],
     "emergency_guidance": "If you or someone you know is experiencing a crisis, please reach out to a crisis helpline immediately..."
   }
   ```

#### 3.5.3 交付物
- [ ] 风险审计模块：`src/audit/risk_checker.py`
- [ ] 危机资源配置：`data/crisis/resources.json`
- [ ] 危机响应模板：`data/crisis/responses.json`
- [ ] 单元测试：`tests/test_risk_audit.py`

#### 3.5.4 验收标准
- 高风险检测准确率 ≥ 98%
- 危机干预响应触发延迟 ≤ 100ms
- 危机资源信息准确、及时更新
- 覆盖主要危机类型（自杀、自伤、暴力等）

---

### 3.6 阶段 2.6: 本地推理服务与 LLM Factory

#### 3.6.1 目标
部署本地 GGUF 模型推理服务，并通过 `LLM Factory` 统一管理不同环境下的模型调用。

#### 3.6.2 实施步骤

1. **GGUF 模型准备**
   - 在 Colab 完成训练及 GGUF 转换
   - 下载模型至本地：`models/psychologist-8b-q4_k_m.gguf`

2. **LLM Factory 构建 (`src/utils/llm_factory.py`)**
   - **功能**：根据环境变量 `LLM_TYPE` 自动切换后端
   - **支持模式**：
     - `MOCK`: 返回固定测试数据（开发用）
     - `CLOUD`: 调用 Deepseek-V3 API（云端推理用）
     - `LOCAL`: 调用本地 `llama-cpp-python`（生产/隐私模式用）

   ```python
   # src/utils/llm_factory.py 伪代码
   class LLMFactory:
       @staticmethod
       def create_llm(llm_type: str = os.getenv("LLM_TYPE", "MOCK")):
           if llm_type == "LOCAL":
               from llama_cpp import Llama
               # 确保使用 chat_format="llama-3" (自动适配 Llama 3 special tokens)
               llm = Llama(
                   model_path="models/psychologist-8b-q4_k_m.gguf",
                   n_gpu_layers=-1,
                   n_ctx=4096,
                   chat_format="llama-3", 
                   verbose=False
               )
               return llm
           elif llm_type == "CLOUD":
               return DeepseekClient(...)
           else:
               return MockLLM()
   ```

3. **本地推理优化 (llama-cpp-python)**
   - **Chat Template 适配**:
     - 严禁手动拼接 `[System]`, `[User]` 等字符串
     - 必须使用 `llm.create_chat_completion(messages=[...])` 接口
     - 示例: `messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
   - **安装 (CUDA版)**：为了利用 GPU 加速，必须安装 CUDA 版本：
     ```bash
     # 确保已安装 CUDA Toolkit 12.x
     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
     ```
   - 显存占用：约 5-6GB (Q4_K_M)，保留余量给系统

#### 3.6.3 硬件要求
| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | GTX 1070 / RTX 2060 (6GB+) | RTX 3060 / 4060 (8GB+) |
| RAM | 16GB | 32GB |
| 存储 | 20GB SSD | 50GB NVMe SSD |

#### 3.6.4 交付物
- [ ] 模型工厂：`src/utils/llm_factory.py`
- [ ] 推理服务：`src/inference/server.py`
- [ ] GGUF 模型文件：`models/*.gguf`
- [ ] 性能测试报告：`reports/inference_benchmark.md`

#### 3.6.5 验收标准
- 单次推理延迟 ≤ 5s (512 tokens)
- 支持至少 2 并发请求
- 服务稳定运行 24 小时无崩溃
- 流式输出 Token 间隔 ≤ 100ms

---

### 3.7 阶段 2.7: 系统集成与主流程

#### 3.7.1 目标
集成所有模块，实现完整的端到端对话流程。

#### 3.7.2 实施步骤

1. **主流程编排**
   ```python
   async def process_user_query(user_input: str, session_id: str) -> str:
       # 1. 安全检查 (Local Embedding)
       safety_result = await safety_gateway.check(user_input)
       if not safety_result.is_safe:
           return safety_result.response

       # 2. RAG 检索
       context = await rag_retriever.retrieve(user_input)

       # 3. 加载会话历史 (获取 10 轮用于 Cloud, 3 轮用于 Local)
       history_long = await memory_store.get_history(session_id, turns=10)
       history_short = await memory_store.get_history(session_id, turns=3)
       user_profile = await memory_store.get_profile(session_id)

       # 4. PII 脱敏
       sanitized_input = pii_redactor.redact(user_input)

       # 5. 生成 Supervisor Prompt 并调用 Deepseek-V3
       prompt_supervisor = prompt_generator.gen_cloud_prompt(
           sanitized_input, context, history_long, user_profile
       )
       # 期望返回纯 JSON
       analysis_json = await deepseek_client.analyze(prompt_supervisor)

       # 6. 风险审计与 Profile 更新
       if analysis_json['risk_level'] == "high":
           return crisis_handler.get_response(analysis_json)
       
       # 更新用户画像
       if analysis_json.get('updated_user_profile'):
           await memory_store.update_profile(session_id, analysis_json['updated_user_profile'])

       # 7. 生成 Agent Prompt (Local)
       prompt_agent = prompt_generator.gen_local_prompt(
           user_input, 
           analysis_json,  # 包含 guidance, technique, core_issue
           context, 
           history_short
       )

       # 8. 本地推理
       response = await local_inference.generate(prompt_agent)

       # 9. 更新记忆
       await memory_store.add(session_id, user_input, response)

       return response
   ```

2. **API 接口设计**
   ```
   POST /api/v1/chat
   {
     "session_id": "uuid",
     "message": "用户消息"
   }

   Response:
   {
     "response": "AI 回复",
     "session_id": "uuid",
     "metadata": {...}
   }
   ```

3. **前端界面（可选）**
   - Gradio / Streamlit 快速原型
   - 或 Web 前端应用

#### 3.7.3 交付物
- [ ] 主流程模块：`src/main.py`
- [ ] API 路由：`src/api/routes.py`
- [ ] 会话管理：`src/session/manager.py`
- [ ] 集成测试：`tests/test_integration.py`
- [ ] 演示界面：`demo/app.py`

#### 3.7.4 验收标准
- 端到端流程可正常运行
- 平均响应时间 ≤ 10s
- 会话上下文保持正确
- 所有边界情况（安全拦截、危机干预、API 降级）处理正确

---

## 4. 项目目录结构

```
psychologist-agent/
├── configs/                    # 配置文件
│   ├── dpo_config.yaml
│   ├── inference_config.yaml
│   └── api_config.yaml
├── data/                       # 数据文件
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后数据
│   ├── dpo/                    # DPO 训练数据
│   ├── knowledge/              # 知识库文档
│   ├── vectordb/               # 向量数据库
│   ├── safety/                 # 安全相关配置
│   └── crisis/                 # 危机干预资源
├── models/                     # 模型文件
│   └── psychologist_qlora/     # QLoRA 权重
├── prompts/                    # Prompt 模板
│   └── prompt_templates.yaml
├── src/                        # 源代码
│   ├── utils/                  # 通用工具
│   │   ├── llm_factory.py      # LLM 工厂模式 (Mock/Cloud/Local)
│   │   └── logging_config.py   # 日志基础配置
│   ├── safety/                 # 安全网关模块
│   ├── rag/                    # RAG 模块
│   ├── memory/                 # 记忆管理模块
│   ├── privacy/                # 隐私保护模块
│   ├── prompt/                 # Prompt 生成模块
│   ├── api/                    # API 客户端
│   ├── audit/                  # 审计模块
│   ├── inference/              # 推理模块
│   ├── session/                # 会话管理
│   └── main.py                 # 主程序入口
├── scripts/                    # 脚本文件
│   ├── data_validation.py
│   ├── generate_baseline_responses.py
│   ├── deepseek_judge.py
│   └── train_dpo.py
├── tests/                      # 测试文件
│   ├── test_safety_gateway.py
│   ├── test_rag.py
│   ├── test_pii_redactor.py
│   ├── test_risk_audit.py
│   └── test_integration.py
├── demo/                       # 演示应用
│   └── app.py
├── logs/                       # 日志文件
├── reports/                    # 报告文件
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略文件（排除数据集、模型权重、敏感配置）
├── README.md                   # 项目说明
└── plan.md                     # 项目计划书
```

---

## 5. 实施时间线（建议）

| 阶段 | 任务 | 建议顺序 |
|------|------|----------|
| 1.1 | 数据准备 | 第 1 阶段 |
| 1.2 | 偏好数据集构建 | 第 2 阶段 |
| 1.3 | DPO 微调 | 第 3 阶段 |
| 2.1 | 安全网关 | 第 4 阶段（可与 2.2 并行） |
| 2.2 | RAG 系统 | 第 4 阶段（可与 2.1 并行） |
| 2.3 | Prompt 与 PII 脱敏 | 第 5 阶段 |
| 2.4 | 云端 API 集成 | 第 5 阶段（可与 2.3 并行） |
| 2.5 | 风险审计与危机干预 | 第 6 阶段 |
| 2.6 | 本地推理服务 | 第 6 阶段（可与 2.5 并行） |
| 2.7 | 系统集成 | 第 7 阶段 |

---

## 6. 风险与注意事项

### 6.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| GPU 显存不足 | 无法训练/推理 | 使用量化、减小 batch size、云端训练 |
| Deepseek API 不稳定 | 服务中断 | 实现降级策略、增加重试机制 |
| DPO 训练效果不佳 | 模型质量差 | 调整超参数、增加数据量、多轮迭代 |
| RAG 检索不准确 | 回答质量下降 | 优化分块策略、改进 embedding 模型 |

### 6.2 伦理与合规风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 用户隐私泄露 | 法律风险、信任损失 | 严格 PII 脱敏、本地化处理 |
| 不当心理建议 | 用户伤害 | 明确免责声明、危机干预机制 |
| 危机情况处理不当 | 严重后果 | 完善安全网关、提供专业资源 |

### 6.3 关键注意事项

1. **免责声明**：系统必须明确告知用户这是 AI 助手，不能替代专业心理咨询
2. **危机干预**：对于自杀、自伤等高风险情况，必须立即引导至专业资源
3. **数据安全**：敏感对话数据必须加密存储，定期清理
4. **持续监控**：上线后需要人工定期审核对话质量和安全性
5. **模型更新**：定期收集反馈，迭代优化模型

---

## 7. 验收清单总览

### Pipeline 1 验收项

- [ ] 数据集准备完成，质量检查通过
- [ ] DPO 数据集构建完成，Judge 评判通过率 ≥ 70%
- [ ] DPO 训练完成，验证集指标达标
- [ ] QLoRA 权重可正常加载和推理

### Pipeline 2 验收项

- [ ] 安全网关模块测试通过
- [ ] RAG 系统检索准确率达标
- [ ] PII 脱敏功能测试通过
- [ ] Deepseek API 集成测试通过
- [ ] 风险审计与危机干预测试通过
- [ ] 本地推理服务性能达标
- [ ] 端到端集成测试通过

### 整体验收

- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] 人工评估对话质量满意
- [ ] 安全边界情况全部覆盖
- [ ] 文档完整（README、API 文档、部署文档）

---

## 8. 附录

### 8.1 参考资源

- [Counsel Chat Dataset](https://huggingface.co/datasets/nbertagnolli/counsel-chat)
- [Llama 3.1 Model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [TRL Library (DPO)](https://github.com/huggingface/trl)
- [PEFT Library (LoRA)](https://github.com/huggingface/peft)
- [LangChain Documentation](https://python.langchain.com/)
- [BGE Embedding Models](https://huggingface.co/BAAI/bge-small-en-v1.5)

### 8.2 API 密钥配置

```bash
# .env 文件（不要提交到版本控制）
DEEPSEEK_API_KEY=sk-a0cf17888de546a18440b34219067218
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 8.3 依赖版本锁定

建议在 `requirements.txt` 中锁定主要依赖版本，确保环境可复现。
