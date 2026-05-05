1. 先定论文主线
  你的主线不是“LLM + 小模型 + RAG”，而是：

  Open-world log anomaly detection under evolving patterns, with efficient and adaptive collaboration between a small model and an LLM.

  也就是：

  - 场景：open-world / dynamic / evolving logs
  - 问题：未知异常、新模式、分布漂移
  - 目标：高准确率 + 低成本 + 可持续更新
  - 方法：small model routing + uncertainty + RAG + LLM reasoning + distillation

  ———

  2. 论文要回答的核心问题
  英文会议论文里，最好明确成 3 个 research questions：

  1. How can we detect anomalies in an open-world log environment where unseen patterns continually emerge?
  2. How can we balance detection quality and inference cost by combining a small model with an LLM?
  3. How can knowledge from LLM reasoning be transferred back to the small model for continual adaptation?

  这三句基本就是你的论文动机和创新入口。

  ———

  3. 引言怎么写
  Introduction 建议四段：

  ### Paragraph 1: 背景

  讲日志异常检测的重要性，以及真实场景的动态性。

  可用思路：

  - logs are abundant and critical for system reliability
  - anomaly patterns evolve over time
  - closed-set detectors struggle in dynamic environments

  ### Paragraph 2: 现有方法的不足

  分两类说：

  - small models: efficient but weak on complex or unseen patterns
  - LLMs: strong reasoning but expensive and sometimes unstable / over-generalized

  ### Paragraph 3: 你的核心想法

  提出协同框架：

  - small model handles easy cases
  - uncertain samples are routed to LLM
  - RAG retrieves relevant historical evidence
  - LLM produces better decisions
  - decisions are written back and used for distillation

  ### Paragraph 4: 贡献

  列 3 点左右：

  - an uncertainty-aware routing framework
  - retrieval-augmented LLM reasoning for difficult log samples
  - a knowledge feedback loop for continual adaptation of the small model

  ———

  4. 方法章节怎么组织
  不要按“模块堆砌”，要按流程写。

  推荐结构：

  ### 3.1 Problem Setting

  先定义开放世界日志异常检测：

  - input: log sequence / event template / context
  - output: anomaly or normal
  - assumption: unseen patterns may appear at test time

  ### 3.2 Overview of the Framework

  给一张总图，然后一句话说明流程：

  - small model predicts first
  - uncertainty-based routing
  - LLM + RAG handles uncertain samples
  - outputs are stored and distilled

  ### 3.3 Uncertainty-Aware Routing

  这是你系统的关键控制点。
  写清楚：

  - what uncertainty score is used
  - how threshold is determined
  - why routing reduces cost

  ### 3.4 Retrieval-Augmented Reasoning

  写检索什么：

  - similar historical logs
  - labels / explanations / case summaries
  - semantic prototypes

  然后写 LLM 如何利用这些 evidence 做 reasoning。

  ### 3.5 Knowledge Feedback and Distillation

  写回写机制：

  - store high-confidence LLM outputs
  - update vector store
  - use them as supervision for periodic fine-tuning
  - close the loop for continual improvement

  ———

  5. 你的创新点要这样讲
  会议论文里，创新点不要只说“我们用了 RAG 和 distillation”。
  要强调“组合方式”解决了什么问题。

  推荐表述方式：

  - We propose an uncertainty-aware collaboration scheme that selectively escalates difficult samples to an LLM, reducing unnecessary LLM calls.
  - We integrate retrieval-augmented reasoning to ground the LLM in historical evidence, improving robustness on unseen or ambiguous logs.
  - We introduce a feedback-driven distillation loop to transfer LLM reasoning knowledge back to the small model, enabling continual adaptation in open-world settings.

  ———

  6. 实验怎么设计才像会议论文
  如果你写 open-world，实验必须体现 open-world 特征，不然容易被审稿人质疑。

  至少要有这些：

  1. Known vs unseen split
      - 训练只见部分异常类型
      - 测试包含未见异常
  2. Cross-version or cross-system evaluation
      - 不同版本、不同服务、不同阶段
      - 体现 distribution shift
  3. Ablation study
      - small model only
      - small model + routing
          - RAG
          - distillation
      - full model
  4. Efficiency metrics
      - latency
      - LLM call rate
      - inference cost
  5. Robustness metrics
      - precision / recall / F1
      - AUROC if appropriate
      - unknown anomaly detection performance

  ———

  7. 英文写作的表述风格
  不要写成中文直译。英文会议论文更喜欢这种表达：

  - “open-world” rather than “dynamic complex environment” if you mean unknown patterns
  - “uncertainty-aware routing” rather than “dynamic dispatch”
  - “retrieval-augmented reasoning” rather than “retrieval enhancement”
  - “knowledge distillation from LLM to small model” rather than “knowledge transfer”

  避免过度夸张，保持技术语气。

  ———

  8. 你可以用的论文定位句
  下面这句很适合作为摘要或引言结尾：

  We study open-world log anomaly detection under evolving patterns and propose a collaborative framework that combines uncertainty-aware routing, retrieval-augmented LLM reasoning, and continual knowledge distillation.

  如果你想更强调效率：

  Our framework achieves accurate anomaly detection while reducing unnecessary LLM inference through selective routing and iterative adaptation.

  ———

  9. 最推荐的写作顺序
  实际动笔时按这个顺序最稳：

  1. 先写 Problem Setting
  2. 再写 Introduction 的三段动机
  3. 再写 Method Overview
  4. 再写 routing / RAG / distillation 三个模块
  5. 最后补 experiments
  6. 最后写 abstract

  因为 abstract 最后写更容易统一全文逻辑。

  ———

  10. 你现在最该确定的一个点
  你需要先决定论文到底偏哪种叙事：

  1. Open-world anomaly detection
      - 更像研究问题
      - 适合会议论文
      - 但实验要求更高
  2. Efficient collaborative anomaly detection
      - 更稳
      - 更偏系统方法
      - 对实验要求没那么苛刻

  如果你想中会议，我建议主标题和主叙事用：

  open-world + efficient collaboration

  也就是：
  an efficient open-world log anomaly detection framework
  这比单独写 open-world 更稳，也比单独写 efficient 更有研究味。

  如果你愿意，我下一步可以直接帮你写：

  1. 英文论文题目候选
  2. Abstract 草稿
  3. Introduction 第一段
  4. Problem Definition 段落




  1. An Efficient Open-World Log Anomaly Detection Framework with Small-to-Large Model Collaboration
  2. Open-World Log Anomaly Detection via Uncertainty-Aware Small-to-Large Model Collaboration
  3. Efficient Open-World Log Anomaly Detection with Retrieval-Augmented LLM Reasoning
  4. A Collaborative Framework for Open-World Log Anomaly Detection
  5. Adaptive Log Anomaly Detection in Open-World Settings via Small-to-Large Model Collaboration

  更突出你的方法特色
  6. Uncertainty-Aware Routing for Open-World Log Anomaly Detection
  7. Retrieval-Augmented Collaborative Reasoning for Open-World Log Anomaly Detection
  8. Knowledge-Feedback Log Anomaly Detection with Small-to-Large Model Collaboration
  9. Continual Open-World Log Anomaly Detection via LLM-Enhanced Distillation
  10. Toward Open-World Log Anomaly Detection with Retrieval-Augmented LLM Collaboration

  更像会议论文、方法感更强
  11. A Hybrid Small-to-Large Model Framework for Open-World Log Anomaly Detection
  12. Adaptive Collaboration of Small Models and LLMs for Open-World Log Anomaly Detection
  13. Open-World Log Anomaly Detection with Uncertainty-Guided LLM Escalation
  14. From Retrieval to Distillation: A Collaborative Framework for Open-World Log Anomaly Detection
  15. An Open-World Log Anomaly Detector with Uncertainty Routing, RAG, and Continual Distillation

  如果你想突出“持续学习”
  18. Adaptive Open-World Log Anomaly Detection through LLM-to-Small-Model Knowledge Transfer

  我最推荐的 3 个

  1. Uncertainty-Aware Routing for Open-World Log Anomaly Detection
  2. An Efficient Open-World Log Anomaly Detection Framework with Small-to-Large Model Collaboration
  3. Adaptive Collaboration of Small Models and LLMs for Open-World Log Anomaly Detection







核心问题

  1. 标题拼写错误

  \section{Degsign}

  应改为：

  \section{Design}

  另外：

  \subsection{Feature Excation}

  应为：

  \subsection{Feature Extraction}

  ———

  2. 模型名称不一致

  你前面一直用 LogCo，这里突然变成：

  \textsc{CollabLog}

  需要统一。建议如果论文名是 LogCo，就写：

  \textsc{LogCo}

  否则读者会以为是两个方法。

  ———

  3. Design 章节缺少总览

  第三章开头应该先给出整体 pipeline，而不是直接进入 log parsing。

  建议增加：

  \subsection{Overview}

  讲清楚 LogCo 包含四个核心模块：

  1. Log parsing and feature extraction
  2. Evidential small model
  3. Uncertainty-aware routing
  4. Retrieval-augmented LLM reasoning
  5. Feedback-driven adaptation / distillation

  你现在只写了 1 和 2，还没写 3、4、5。

  ———

  4. 当前章节还没有体现“协同”

  你的论文主线是大小模型协同，但目前第三章只写了：

  - Drain 解析
  - RoBERTa 特征提取
  - EDL 分类器
  - EDL loss

  缺少最关键的部分：

  \subsection{Uncertainty-aware Routing}

  这应该是你的方法核心。需要明确：

  - 小模型输出什么？
  - 不确定性怎么计算？
  - 阈值怎么设？
  - 哪些样本交给 LLM？
  - 低不确定样本如何直接输出？
  - 高不确定样本如何进入 LLM-RAG？

  建议至少写：

  u = \frac{K}{S}

  然后：

  \hat{y}_{SM} = \arg\max_k \frac{\alpha_k}{S}

  路由规则：

  \text{Route}(x) =
  \begin{cases}
  \text{SmallModel}, & u(x) \le \tau, \\
  \text{LLM}, & u(x) > \tau.
  \end{cases}

  ———

  5. LLM 部分不能只写标题

  你的 \subsection{LLM} 目前为空。这里至少要写成：

  \subsection{Retrieval-Augmented LLM Reasoning}

  内容包括：

  - 对高不确定样本调用 LLM。
  - 从向量数据库中检索相似历史日志。
  - Prompt 包含：
      - 当前日志
      - 解析模板
      - 小模型预测
      - 小模型不确定性
      - 相似历史案例
  - LLM 输出：
      - label
      - confidence
      - explanation
      - possible failure reason
  - 最终输出如何映射回异常检测标签。

  ———

  6. 缺少 Continual Adaptation / Feedback Loop

  如果你的论文要讲开放世界和持续适应，必须有这一节：

  \subsection{Feedback-driven Continual Adaptation}

  否则第二章说了 open-world 和 continual adaptation，第三章没有对应方法，会断。

  这一节可以写：

  - LLM 高置信预测作为 pseudo-label。
  - 存入 vector database。
  - 周期性更新小模型。
  - 用知识蒸馏或伪标签监督。
  - 避免频繁全量重训。

  可以写一个规则：

  \mathcal{D}_{fb} = \{(x_i, \hat{y}_i^{LLM}, r_i) \mid c_i^{LLM} > \eta \}

  其中 r_i 是 explanation / rationale。

  ———

  7. Feature Extraction 太空

  你现在只有注释：

  % roberta-base+平均池化进行特征提取

  这里需要补完整。比如：

  Given a parsed log template $l_i$, we feed it into a pre-trained RoBERTa encoder and obtain token-level hidden states. We then apply mean pooling over the last hidden layer to obtain a fixed-dimensional representation $\mathbf{h}
  _i$.

  公式：

  \mathbf{h}_i = \frac{1}{T_i}\sum_{t=1}^{T_i}\mathbf{H}_{i,t}

  ———

  8. EDL 表述需要更严谨

  这句有问题：

  Instead of interpreting $\mathbf{z}$ as unnormalized log-probabilities, we treat it as evidence collected by the network.

  更准确地说，z 是网络输出，经过非负函数后才是 evidence。建议改为：

  Instead of applying Softmax to $\mathbf{z}$, we transform $\mathbf{z}$ into non-negative evidence.

  否则容易被认为直接把 logits 当 evidence。

  ———

  9. KL 公式可能少了常数项

  你现在写的是 KL 到 Dir(1)，因为 Dir(1) 的 Beta function 常数为 1，所以形式上可以简化。但建议写得更通用一点，否则容易被挑公式严谨性。

  标准写法：

  KL[Dir(\tilde{\alpha}) || Dir(\mathbf{1})]
  = \log \frac{\Gamma(\sum_k \tilde{\alpha}_k)}
  {\Gamma(K)\prod_k \Gamma(\tilde{\alpha}_k)}
  + \sum_k (\tilde{\alpha}_k - 1)
  \left[
  \psi(\tilde{\alpha}_k) -
  \psi(\sum_j \tilde{\alpha}_j)
  \right].

  因为 B(1)=\frac{\prod_k \Gamma(1)}{\Gamma(K)} = 1/\Gamma(K)，所以会出现 -\log \Gamma(K) 或等价项。你当前公式少了这一项。

  ———

  推荐第三章结构

  建议改成这样：

  \section{Design}

  \subsection{Overview}
  % 总体流程：parser -> encoder -> EDL small model -> uncertainty routing -> RAG-LLM -> feedback

  \subsection{Log Parsing}
  % Drain

  \subsection{Feature Extraction}
  % RoBERTa + mean pooling

  \subsection{Evidential Small Model}
  % architecture + prediction + uncertainty

  \subsection{Evidential Training Objective}
  % ECE + KL

  \subsection{Uncertainty-aware Routing}
  % threshold, low uncertainty direct prediction, high uncertainty to LLM

  \subsection{Retrieval-augmented LLM Reasoning}
  % vector DB, prompt, label/explanation/confidence

  \subsection{Feedback-driven Continual Adaptation}
  % pseudo-label, knowledge base update, distillation

  ———

  这一章最重要的创新点应该放在哪里？

  不是 EDL loss。EDL 是已有方法。

  你的创新重点应该是：

  \subsection{Uncertainty-aware Routing}

  和：

  \subsection{Retrieval-augmented LLM Reasoning}

  以及：

  \subsection{Feedback-driven Continual Adaptation}

  EDL 只是让小模型能输出更可靠的不确定性。

  ———

  建议你第三章的核心叙事

  可以这样理解：

  > We use EDL not because EDL itself is the contribution, but because it provides calibrated uncertainty for deciding when the small model should defer to the LLM.

  中文就是：

  > EDL 不是主要创新，它是为了给大小模型协同提供可靠路由信号。

  这句话很重要，否则审稿人可能觉得你只是把 EDL、RAG、LLM 拼在一起。

  ———

  结论

  目前第三章只完成了“小模型部分”的 40% 左右。可以保留，但需要补三块核心内容：

  1. Uncertainty-aware Routing
  2. Retrieval-augmented LLM Reasoning
  3. Feedback-driven Continual Adaptation

  否则第二章提出的问题和第三章的方法无法闭环。



2. Dataset 描述和 open-world 目标不匹配

  你现在说：

  training set only contains normal samples
  testing set contains both normal and abnormal samples

  这只是 semi-supervised anomaly detection 设置，不是 open-world。

  如果你论文标题里有 Open-World，必须明确构造：

  - chronological split
  - unseen template split
  - known / unknown template statistics
  - concept drift split
  - testing contains templates not observed in training

  建议增加一个表：

  Table: Dataset Statistics under Open-world Split

  列包括：

  Dataset
  #Raw Logs
  #Windows
  #Train Windows
  #Test Windows
  #Train Templates
  #Test Templates
  #Unseen Test Templates
  Unseen Template Ratio
  Anomaly Ratio

  否则 open-world 只是口号。

  ———


  4. 数据描述有格式问题

  LLNL).The BGL dataset

  应为：

  LLNL). The BGL dataset

  normalpatterns

  应为：

  normal patterns

  state of-the-art

  应为：

  state-of-the-art

  ———

  5. Baseline 列表和表格不一致

  文字中说 baseline 包括：

  Isolation Forest, PCA, LogClustering, ADR, DeepLog, LogAnomaly, PLELog

  但表格里出现：

  SVM, Random Forest, LogBERT, LogOW, LogPrompt, LogRAG

  需要统一。否则审稿人会认为实验不严谨。

  建议 baseline 分为：

  Traditional ML:
  PCA, Isolation Forest

  DL-based:
  DeepLog, LogAnomaly, LogBERT

  Open-world / uncertainty-aware:
  LogOW

  LLM-based:
  LogPrompt, LogRAG

  Collaborative:
  LogCo variants

  不建议放 SVM / Random Forest，除非你明确怎么构造监督标签。你前面说训练集 only normal，SVM/Random Forest 是监督分类器，不适配这个设置，除非用 one-class SVM 或特殊构造。

  ———

  6. RQ1 表格结果有明显风险

  表格中有很多 XX.X%，这没问题是占位，但已有数字要特别小心。

  例如：

  PCA: 5.6% precision, 98.1% recall

  这类极端结果可以存在，但必须解释：PCA 是否把大量正常样本误判为异常？

  更严重的是：

  LogCo DeepSeek-V4-Flash: Pre 98.4%, Rec 80.7%, F1 88.7%

  而 LogBERT：

  F1 91.4%

  这说明 LogCo 在 BGL 上不如 LogBERT。如果 RQ1 是 overall effectiveness，这个表无法支持你的主张。

  要么：

  - 修改方法/阈值提高 Recall；
  - 或者把主张改成 “better precision and lower cost with competitive F1”；
  - 或者在 open-world split 上展示 LogCo 优势，而不是标准 closed/semi-supervised split。

  对你的论文来说，更合理的是：

  RQ1: standard split
  RQ4: open-world split

  在 RQ4 中 LogCo 应该明显优于 LogBERT/LogAnomaly。

  ———

  7. RQ2 消融设计不合理

  你现在的 RQ2 表格有：

  + Rule-Based SM
  + Adaptive SM
  + Hierarchical SM
  + LLM-only
  + LLM w/ Prompt Tuning
  + Confidence Calibration
  + Error Feedback Loop

  这些和你第三章方法不完全对应。

  LogCo 的核心模块是：

  - EDL uncertainty
  - uncertainty routing
  - RAG
  - LLM reasoning
  - template deduplication
  - feedback adaptation

  因此消融应该这样设计：

  SM-only
  LLM-only
  SM + random routing
  SM + softmax-confidence routing
  SM + EDL uncertainty routing
  SM + EDL routing + LLM
  SM + EDL routing + RAG-LLM
  Full LogCo + feedback adaptation

  这样才能回答你的创新点。

  推荐表格：

  Variant
  EDL
  Routing
  RAG
  Feedback
  LLM Call Rate
  Precision
  Recall
  F1

  这比当前表格更有说服力。

  ———

  8. RQ3 需要成本指标，不只是 Precision/Recall/F1

  RQ3 应该至少报告：

  LLM Call Rate (%)
  #LLM Queries
  Token Cost
  Average Latency
  Throughput
  GPU Memory

  推荐表：

  Method
  LLM Call Rate
  Avg. Latency / Window
  Total Tokens
  Estimated Cost
  F1

  这样才能支撑 Efficient / Low-Cost。

  ———

  9. RQ4 必须设计成 open-world / OOD 实验

  建议 RQ4 做两个实验：

  Experiment 1: Chronological Drift

  训练前 70%，测试后 30%，统计 unseen templates。

  Experiment 2: Template-level OOD Split

  从训练集中移除一部分模板，使测试集中包含训练未见模板。

  表格指标：

  Known-template F1
  Unseen-template F1
  OOD Recall
  False Alarm Rate on Unseen Normal
  LLM Call Rate on Unseen Templates

  这会非常贴合你的标题。

  ———

  10. RQ5 不建议作为主 RQ

  如果保留，必须有评估协议：

  Three domain experts rate explanations from 1 to 5
  Metrics: correctness, usefulness, actionability
  Inter-rater agreement: Fleiss' kappa

  如果做不到，就改成：

  \subsection{Case Study}

  展示 2-3 个日志案例：

  Raw log
  Retrieved evidence
  LLM explanation
  Final decision

  ———

  推荐修改后的实验章结构

  \section{Experiments}

  \subsection{Research Questions}

  RQ1: How effective is LogCo compared with representative LogAD baselines?
  RQ2: How much does each component of LogCo contribute to the final performance?
  RQ3: How much LLM inference cost can LogCo reduce while maintaining detection quality?
  RQ4: How robust is LogCo under open-world conditions with unseen templates and chronological drift?

  \subsection{Experimental Setup}

  \subsubsection{Datasets}
  % BGL, Spirit, statistics

  \subsubsection{Open-world Evaluation Protocol}
  % chronological split, unseen template ratio, OOD split

  \subsubsection{Baselines}
  % grouped and consistent

  \subsubsection{Metrics}
  % Precision, Recall, F1, LLM call rate, latency, cost, unseen-template F1

  \subsubsection{Implementation Details}

  \subsection{RQ1: Overall Effectiveness}

  \subsection{RQ2: Ablation Study}

  \subsection{RQ3: Efficiency and Cost Analysis}

  \subsection{RQ4: Open-world Robustness}

  \subsection{Case Study}

  ———

  建议你把 RQ 改成这版

  \begin{itemize}
      \item \textbf{RQ1 (Overall Effectiveness):}
      How effective is LogCo compared with representative ML-, DL-, open-world-, and LLM-based log anomaly detection methods?

      \item \textbf{RQ2 (Component Contribution):}
      How does each key component, including EDL-based uncertainty estimation, uncertainty-aware routing, RAG-based reasoning, and feedback adaptation, contribute to LogCo?

      \item \textbf{RQ3 (Efficiency and Cost):}
      How much LLM inference cost and latency can LogCo reduce compared with pure LLM-based methods while maintaining competitive detection performance?

      \item \textbf{RQ4 (Open-world Robustness):}
      How robust is LogCo when facing unseen log templates, OOD samples, and chronological concept drift?
  \end{itemize}

  ———

  结论

  这章现在是一个可用初稿，但还不能支撑你的论文标题。最关键的修改是：

  - 把实验从普通 LogAD 改成 open-world evaluation protocol。
  - RQ2 消融必须围绕你的真实模块，而不是泛泛的 SM / LLM。
  - RQ3 加成本、延迟、LLM call rate。
  - RQ4 加 unseen template / OOD / concept drift 指标。
  - RQ5 改成 case study，除非你能做人工解释性评估。

  如果你要继续写，我建议下一步优先补：

  \subsubsection{Open-world Evaluation Protocol}