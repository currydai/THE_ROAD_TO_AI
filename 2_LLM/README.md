# 大语言模型（LLM）教程目录

---

## 第一部分：基础与概念

1. 从语言模型到大语言模型
   - 语言模型的定义与目标函数
   - N-gram、RNN 到 Transformer 的演化
   - 自回归（AR）与自编码（AE）的区别
   - GPT 与 BERT 的基本思想比较

2. Transformer 核心机制
   - 注意力机制（Self-Attention, Multi-Head Attention）
   - 残差连接与层归一化（Residual & LayerNorm）
   - 位置编码（Sinusoidal, RoPE, ALiBi）
   - 编码器与解码器结构（Encoder-Decoder, Decoder-only）

3. 训练目标与损失函数
   - 自回归语言建模（Causal LM Loss）
   - 掩码语言建模（Masked LM Loss）
   - Tokenization（BPE, SentencePiece, tiktoken）
   - 优化与采样策略（Teacher Forcing, Top-k, Top-p）

4. Scaling Law 与 Emergent Ability
   - 模型规模、数据量、计算量的幂律关系
   - Chinchilla Scaling Law
   - Tokenization（BPE, SentencePiece, tiktoken）
   - 涉及推理、记忆、组合泛化的“涌现能力”案例
---

## 第二部分：预训练与微调

4. 预训练（Pretraining）
   - 大规模语料构建（CC, The Pile, C4, RefinedWeb）
   - 数据清洗与去重（Deduplication, Filtering）
   - 并行训练策略（Data / Model / Pipeline / Tensor）
   - 混合精度与梯度检查点（AMP, Checkpointing）

5. 微调（Fine-tuning）
   - 监督微调（Supervised Fine-Tuning, SFT）
   - 指令微调（Instruction Tuning, Chat Tuning）
   - Prompt 模板与系统角色（ChatML, Alpaca Format）
   - 参数高效微调（PEFT：LoRA, QLoRA, Prefix-Tuning）

6. 对齐（Alignment）
   - 人类反馈强化学习（RLHF: Reward Model + PPO）
   - 直接偏好优化（DPO）
   - Constitutional AI 与自对齐（Self-Alignment）
   - 安全性与价值观对齐（Safety, Bias, Toxicity）

---

## 第三部分：系统与推理

7. 模型架构与参数设计
   - 主流架构对比（GPT, LLaMA, Qwen, Mistral, Mixtral）
   - 层归一化方式（LayerNorm, RMSNorm）
   - 激活函数（GELU, SwiGLU）
   - FlashAttention 与 KV Cache 优化

8. 推理与部署（Inference）
   - 推理流程（Token-by-Token Generation）
   - Speculative Decoding 与 KV Cache 复用
   - 模型量化（4bit, 8bit, GPTQ, AWQ）
   - 高效推理框架（vLLM, TGI, Exllama）

9. 训练框架与工程化实现
   - Hugging Face Transformers 全流程
   - DeepSpeed / Megatron-LM / ColossalAI
   - Checkpoint 合并、转换与裁剪
   - 分布式训练与监控工具（W&B, TensorBoard）

---

## 第四部分：应用与生态

10. 提示工程（Prompt Engineering）
    - Zero-shot, Few-shot, Chain-of-Thought
    - ReAct（Reason + Act）与 Tree-of-Thoughts
    - System Prompt、上下文窗口与 Token 限制
    - Prompt 优化与自动生成

11. 检索增强生成（RAG）
    - 检索增强原理与信息注入机制
    - 文档分块与向量嵌入（Embeddings）
    - 向量数据库（FAISS, Milvus, Chroma）
    - RAG 与 Fine-tuning 的结合方式

12. 多智能体系统（Multi-Agent System）
    - 智能体结构：Planner / Executor / Memory / Tool
    - LangChain、CrewAI、AutoGPT 框架
    - 自主任务规划与多代理协作
    - 实际应用：科研、代码生成、地球物理解释

13. 工具与函数调用（Function Calling）
    - Function Calling 机制（OpenAI, Anthropic）
    - ReAct + Tool + Memory 混合架构
    - WebAgent / OS-Agent 实现思路
    - 外部API与插件系统集成

---

## 第五部分：前沿与趋势

14. 多模态大模型（MLLM）
    - 文本 + 图像：CLIP、BLIP、GPT-4V
    - 文本 + 音频 / 视频 / 传感器融合
    - 多模态输入的对齐与适配器设计

15. 模型压缩与蒸馏
    - 剪枝（Pruning）、量化（Quantization）
    - 知识蒸馏（Knowledge Distillation）
    - TinyLLM 与边缘设备部署

16. 模型评测与可解释性
    - Benchmark：MMLU, GSM8K, BIG-Bench
    - 评测维度：知识、推理、安全、价值观
    - 注意力可视化与 Attribution 分析

17. 自主智能与未来发展
    - World Model 与 Memory-Augmented LM
    - Self-Improving 与 Continual Learning
    - Agentic Systems（MetaGPT, Voyager）
    - AI for Science / Materials / Geophysics 应用前景

---
