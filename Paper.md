# 摘要
日志异常检测是保障大规模软件系统稳定性与安全性的关键技术，但在真实场景中普遍面临两类挑战：一是异常样本稀缺
与分布漂移导致小模型泛化受限，二是仅依赖单阶段判别难以兼顾检测性能与推理成本。本文提出一种面向日志分析的级
联检测框架 LogCo，将基于证据深度学习（EDL）的小模型与基于检索增强生成（RAG）的大语言模型（LLM）有机结合。第
一阶段，小模型在全测试集上输出异常分数与不确定度估计；第二阶段，仅对高不确定样本触发 LLM 二次判别，并利用向
量检索返回的带标签参考日志提供可解释证据。为降低调用开销，本文在二次检测中引入基于日志模板归一化的去重策
略，将唯一日志级推理结果映射回样本级评估。实验流程支持高不确定子集与全测试集两种评测范围，并提供灵活的不确
定样本决策策略。结果表明，不确定度与错误率具有显著相关性，级联机制能够在有限 LLM 成本下有效提升总体检测鲁棒
性与实用性。本文框架具备良好的工程可复现性，可为生产环境中的高性价比日志异常检测提供参考。
# Introduction
Log data is a primary source for monitoring, diagnosing, and maintaining modern large-scale software systems. As cloud-native and microservice architectures continue to evolve, log streams become increasingly noisy, heterogeneous, and dynamic, making anomaly detection both essential and challenging. In practice, log anomaly detection is often deployed under an open-world setting: new anomaly patterns, previously unseen normal behaviors, and distribution shifts may emerge after deployment, which limits the reliability of detectors trained on fixed patterns.

Existing approaches can be broadly grouped into two categories. Small models, including traditional machine learning and deep learning-based detectors, are attractive for their efficiency and scalability, but they typically rely on relatively fixed representations and struggle with ambiguous, rare, or evolving patterns. In contrast, large language models (LLMs) exhibit strong contextual reasoning and generalization ability, making them promising for semantic log analysis. However, directly applying an LLM to all incoming logs is computationally expensive and operationally impractical, and pure prompting or fine-tuning alone does not provide a stable mechanism for balancing accuracy, latency, and cost in long-running systems.

To address this trade-off, we study an efficient open-world log anomaly detection problem and propose LogCo, a collaborative framework that combines a small model with an LLM through uncertainty-aware routing and retrieval-augmented reasoning. The key idea is to let the small model handle easy samples efficiently, while escalating high-uncertainty cases to the LLM for deeper analysis. Specifically, the small model first estimates both the anomaly score and prediction uncertainty using evidential deep learning (EDL). The uncertainty score then serves as a routing signal: low-uncertainty samples are directly classified by the small model, whereas high-uncertainty samples are sent to the LLM.

For the escalated samples, LogCo further equips the LLM with retrieval-augmented generation (RAG). A vector database stores historical logs, labels, and structured annotations, allowing the LLM to retrieve semantically related evidence before making a decision. This retrieval step grounds the LLM in concrete historical cases and helps reduce over-generalized or unsupported predictions. In addition, the vector database is only accessed during LLM reasoning, which preserves a clear separation between efficient first-stage inference and knowledge-intensive second-stage analysis.

Beyond inference-time collaboration, LogCo introduces a feedback-driven learning loop for continual adaptation. High-confidence outputs from the LLM are stored as structured pseudo-labels and reused to update the knowledge base, while also serving as supervision signals for periodically distilling the small model. This design allows the system to gradually transfer knowledge from the LLM back to the efficient detector, reducing future LLM usage while improving coverage of emerging patterns.

Our contributions are summarized as follows:

1. We formulate open-world log anomaly detection as an efficient collaborative inference problem and propose an uncertainty-aware routing mechanism that dynamically allocates samples between a small model and an LLM.
2. We design a retrieval-augmented reasoning module for difficult log samples, enabling the LLM to make evidence-grounded and context-aware predictions.
3. We introduce a feedback-driven distillation framework that continuously transfers LLM knowledge back to the small model, supporting continual adaptation in evolving log environments.
4. We conduct extensive experiments showing that LogCo substantially reduces LLM usage while maintaining or improving detection performance compared with strong baselines.

Overall, this work provides a practical paradigm for open-world log anomaly detection by combining efficient first-stage filtering, LLM-based reasoning on difficult cases, and continual knowledge transfer across model scales.
