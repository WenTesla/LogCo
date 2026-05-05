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

# Background and Motivation

## Existing Paradigms for Log-based Anomaly Detection

### ML/DL-based Methods

To reduce the heavy manual effort required for log analysis, a large body of prior work has investigated machine learning (ML)-based log anomaly detection methods. These methods typically extract structured features from raw logs and then train classifiers or statistical models, such as PCA-based detectors and SVM-based classifiers, to distinguish normal system behavior from abnormal behavior. Compared with manual rule construction, ML-based methods improve automation and scalability, but their effectiveness often depends on handcrafted features and relatively stable log distributions.

Deep learning (DL)-based methods further improve detection performance by learning complex representations from log data. Existing DL approaches model word-level semantics, log template semantics, and event sequence dependencies, enabling them to capture richer contextual patterns than traditional ML methods. As a result, DL-based methods often achieve strong detection accuracy under standard benchmark settings. However, their practical deployment in industrial environments is still limited by two major issues. First, most DL models lack interpretability. They usually output anomaly labels or scores without providing human-readable explanations, making it difficult for operators to understand the cause of a detection result. Second, DL models are vulnerable to out-of-distribution (OOD) data and concept drift. In modern IT systems, software updates, configuration changes, workload shifts, and service evolution continuously introduce novel log patterns. Static DL models trained on limited historical data may fail to recognize these new patterns and can produce overconfident but incorrect predictions.

### LLM-based Methods

Recently, large language models (LLMs) have been introduced into log anomaly detection. Unlike conventional DL models that rely heavily on predefined templates or fixed representations, LLMs can directly interpret raw log messages and generate natural language rationales for their predictions. Their strong language understanding and reasoning capabilities make them particularly attractive for analyzing ambiguous logs, unseen templates, and complex failure symptoms.

Despite their advantages in semantic understanding and explainability, LLM-based methods face serious efficiency and deployment challenges. Since LLMs usually contain billions of parameters, their inference requires substantial computational resources and introduces high latency. Directly applying an LLM to every incoming log is therefore difficult to sustain in real-time scenarios with high log throughput. Commercial API-based LLM services can reduce the burden of local deployment, but they introduce additional privacy and security risks because system logs may contain sensitive operational information. Moreover, long-term API-based inference can lead to prohibitive operational costs. Therefore, although LLMs are powerful semantic reasoners, using them as standalone log anomaly detectors is often impractical in production environments.

### Collaborative DL-LLM Methods

To combine the efficiency of DL models and the explainability of LLMs, recent studies have started to explore collaborative paradigms for log anomaly detection. In such frameworks, a lightweight DL model is typically used as a first-stage detector, while the LLM is invoked only for selected samples that require deeper semantic analysis. This design can reduce unnecessary LLM inference while preserving the ability to obtain explanations and reasoning for difficult cases.

However, existing collaborative methods still face two key limitations. First, the decision of when to invoke the LLM is often based on heuristic confidence scores derived from raw logits or softmax outputs. These scores are not necessarily calibrated and may not faithfully reflect the true prediction risk of the small model. Deep models are known to be overconfident on OOD samples, which is particularly problematic in evolving log environments. Second, the relationship between model confidence and actual misclassification risk is often assumed rather than explicitly validated. Without such validation, low-confidence routing may not reliably identify error-prone or hard-to-classify samples, potentially wasting LLM resources on easy cases while leaving risky samples to the small model. These limitations motivate a more robust, uncertainty-aware, and empirically validated collaboration mechanism.

| Paradigm | Explainability | Efficiency | Reasoning Capability | OOD Robustness | Continual Adaptation | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| ML/DL-based | Limited | High | Moderate | Low to Moderate | Limited | Low |
| LLM-based | High | Low | High | High | Moderate | High |
| LogCo (Ours) | High | High | High | High | High | Moderate |

## Open-world Challenges in Log Anomaly Detection

Most existing log anomaly detection methods are developed under a closed-world assumption, where the training and test data are expected to follow similar distributions and the set of log patterns is assumed to be relatively fixed. This assumption does not hold in real-world operation environments. Logs are continuously affected by system upgrades, configuration changes, service scaling, new user behaviors, and evolving fault modes. As a result, new normal patterns, unseen anomaly types, and shifted event dependencies may appear after deployment.

This open-world setting introduces three challenges. First, the detector must recognize unknown or unseen log patterns instead of forcing them into known categories. Second, the model must remain reliable under distribution shifts, where historical decision boundaries may no longer be valid. Third, the system should continuously absorb useful knowledge from newly observed logs without requiring frequent full retraining or extensive manual annotation. These challenges require log anomaly detection systems to go beyond static classification and support efficient, interpretable, and continual adaptation.

## Motivation: Uncertainty-guided Large-Small Model Collaboration

The fundamental challenge is not simply how to combine a small DL model with an LLM, but how to decide when the lightweight model is trustworthy and when the LLM should be invoked under open-world distribution shifts. Inspired by the ``System 1'' and ``System 2'' view of fast and slow thinking, LogCo follows a hybrid design: the small model acts as an efficient first-stage detector for routine samples, while the LLM serves as a slower but stronger reasoner for uncertain and novel cases.

Our motivation is threefold.

- Efficiency requires selective LLM invocation. Since log streams are often high-throughput, invoking an LLM for every sample is computationally expensive and operationally impractical. A lightweight model should handle easy and familiar samples, while the LLM should be reserved for samples that are difficult, ambiguous, or potentially novel.

- Open-world environments require uncertainty awareness. In evolving systems, OOD samples, unseen log templates, and concept drift can make the small model unreliable. Prediction uncertainty provides a natural signal for identifying such risky cases and routing them to stronger semantic reasoning modules.

- Reliable collaboration requires calibrated and validated confidence. Raw softmax scores or logits may produce overconfident predictions and cannot be directly treated as reliable risk estimates. Therefore, the collaboration mechanism should explicitly estimate uncertainty and validate whether high-uncertainty samples are indeed more likely to be misclassified. This validation is essential for making LLM invocation both effective and cost-efficient.

Based on these observations, LogCo adopts an uncertainty-aware cascaded architecture. The small model first produces both anomaly predictions and uncertainty estimates. Low-uncertainty samples are handled locally to preserve efficiency, while high-uncertainty samples are escalated to the LLM. For escalated samples, retrieval-augmented reasoning provides relevant historical evidence, enabling the LLM to generate more grounded and explainable decisions. The resulting high-confidence LLM outputs can further be stored as structured knowledge and used for future adaptation of the small model. In this way, LogCo aims to achieve efficient, explainable, and continually adaptive log anomaly detection in open-world environments.

# Design

We now present the design of \textsc{LogCo}, an uncertainty-guided collaborative framework for open-world log anomaly detection. The key idea is to use a lightweight evidential small model for efficient first-stage detection and selectively invoke an LLM for uncertain cases that require stronger semantic reasoning. The framework consists of five components: log parsing, feature extraction, evidential small-model prediction, uncertainty-aware routing, retrieval-augmented LLM reasoning, and feedback-driven continual adaptation.

## Overview

Given a stream of raw logs $\mathcal{X}=\{x_i\}_{i=1}^{N}$, \textsc{LogCo} first converts each raw log into a structured representation through log parsing. The parsed log template is then encoded by a pre-trained language model to obtain a dense semantic representation. On top of this representation, an evidential classifier predicts both the anomaly label and the epistemic uncertainty of the prediction. For low-uncertainty samples, \textsc{LogCo} directly accepts the small model's prediction to preserve efficiency. For high-uncertainty samples, \textsc{LogCo} invokes a retrieval-augmented LLM reasoning module, which retrieves semantically similar historical logs and uses them as evidence for the final decision. The high-confidence outputs produced by the LLM are further stored as structured knowledge and can be reused for future retrieval and small-model adaptation.

Formally, for each log instance $x_i$, the framework produces a final prediction:

$$
\hat{y}_i =
\begin{cases}
\hat{y}_i^{SM}, & u_i \le \tau, \\
\hat{y}_i^{LLM}, & u_i > \tau,
\end{cases}
$$

where $\hat{y}_i^{SM}$ is the prediction of the small model, $\hat{y}_i^{LLM}$ is the prediction of the LLM reasoning module, $u_i$ is the uncertainty estimated by the evidential classifier, and $\tau$ is the routing threshold. This cascaded design reduces unnecessary LLM calls while retaining the ability to handle ambiguous, novel, and OOD logs.

## Log Parsing

Raw logs are semi-structured or unstructured text records that usually contain timestamps, severity levels, component names, identifiers, and free-form messages. Instance-specific parameters, such as block IDs, IP addresses, request IDs, and file paths, may introduce noise and hinder downstream anomaly detection. Therefore, before model inference, \textsc{LogCo} applies log parsing to transform raw logs into structured templates.

In this work, we adopt Drain for log parsing. Drain constructs a fixed-depth parse tree and groups similar log messages into templates by replacing variable tokens with wildcards. Given a raw log $x_i$, the parser produces a normalized template $t_i$ and optional structured fields:

$$
t_i = \mathrm{Parser}(x_i).
$$

The parsed template preserves the stable semantic pattern of the log while removing instance-specific parameters. This normalization improves representation consistency and also enables deduplication before LLM invocation, reducing repeated reasoning over semantically identical logs.

## Feature Extraction

After log parsing, \textsc{LogCo} extracts semantic features from each parsed template. Given a template $t_i$, we feed it into a pre-trained RoBERTa encoder and obtain token-level hidden states from the last transformer layer:

$$
\mathbf{H}_i = [\mathbf{h}_{i,1}, \mathbf{h}_{i,2}, \dots, \mathbf{h}_{i,T_i}],
$$

where $T_i$ is the number of tokens and $\mathbf{h}_{i,j} \in \mathbb{R}^{d}$ is the hidden representation of the $j$-th token. We then apply mean pooling to obtain a fixed-dimensional representation:

$$
\mathbf{h}_i = \frac{1}{T_i}\sum_{j=1}^{T_i}\mathbf{h}_{i,j}.
$$

The resulting vector $\mathbf{h}_i$ is used as the input to the evidential classifier. Compared with sparse template IDs or manually designed features, RoBERTa-based semantic representations can better capture lexical and contextual similarities among logs, which is useful for both known-pattern detection and unseen-template generalization.

## Evidential Small Model

The small model is designed to provide both efficient anomaly prediction and uncertainty estimation. Traditional classifiers usually apply a Softmax function to logits and produce point estimates of class probabilities. However, Softmax confidence is often poorly calibrated and may be overconfident on OOD samples. To obtain a more explicit uncertainty signal, \textsc{LogCo} adopts evidential deep learning (EDL).

Let $\mathbf{h}_i \in \mathbb{R}^{d}$ denote the feature representation extracted by the RoBERTa encoder. The feature is fed into a multi-layer perceptron (MLP) with batch normalization, ReLU activation, and dropout:

$$
\mathbf{z}_i = f_{\theta}(\mathbf{h}_i), \quad \mathbf{z}_i \in \mathbb{R}^{K},
$$

where $K$ is the number of classes. Instead of applying Softmax to $\mathbf{z}_i$, we transform the output into non-negative evidence:

$$
\mathbf{e}_i = \mathrm{ReLU}(\mathbf{z}_i).
$$

The evidence is then used to parameterize a Dirichlet distribution:

$$
\alpha_{i,k} = e_{i,k} + 1, \quad k=1,2,\dots,K.
$$

The Dirichlet strength is defined as:

$$
S_i = \sum_{k=1}^{K}\alpha_{i,k}.
$$

The expected class probability is:

$$
p_{i,k} = \frac{\alpha_{i,k}}{S_i},
$$

and the small model prediction is:

$$
\hat{y}_i^{SM} = \arg\max_{k} p_{i,k}.
$$

The epistemic uncertainty is quantified as:

$$
u_i = \frac{K}{S_i}.
$$

Intuitively, if the model collects strong evidence for one or more classes, $S_i$ becomes large and the uncertainty decreases. If the model lacks sufficient evidence, $S_i$ remains small and the uncertainty increases. This uncertainty score serves as the routing signal for deciding whether a sample should be handled by the small model or escalated to the LLM.

## Evidential Training Objective

To train the evidential classifier, we optimize an objective based on the Dirichlet distribution. Given a training sample with one-hot label $\mathbf{y}_i$, the expected cross-entropy loss is:

$$
\mathcal{L}_{ECE}^{(i)}
= \sum_{k=1}^{K} y_{i,k}\left(\log S_i - \log \alpha_{i,k}\right).
$$

This term encourages the model to assign high evidence to the correct class. However, only minimizing the expected cross-entropy may still allow the model to assign misleading evidence to incorrect classes. Therefore, we introduce a KL regularization term that penalizes unnecessary evidence for false classes.

We first define the modified Dirichlet parameter:

$$
\tilde{\boldsymbol{\alpha}}_i
= \mathbf{y}_i + (1-\mathbf{y}_i)\odot \boldsymbol{\alpha}_i,
$$

which removes the evidence of the true class while preserving the evidence assigned to incorrect classes. The KL regularization term is defined as:

$$
\mathcal{L}_{KL}^{(i)}
= KL\left[\mathrm{Dir}(\tilde{\boldsymbol{\alpha}}_i)\,\|\,\mathrm{Dir}(\mathbf{1})\right],
$$

where $\mathrm{Dir}(\mathbf{1})$ denotes the uniform Dirichlet distribution. Its expanded form is:

$$
\begin{aligned}
\mathcal{L}_{KL}^{(i)}
&= \log \frac{\Gamma(\sum_{k=1}^{K}\tilde{\alpha}_{i,k})}
{\Gamma(K)\prod_{k=1}^{K}\Gamma(\tilde{\alpha}_{i,k})} \\
&\quad + \sum_{k=1}^{K}(\tilde{\alpha}_{i,k}-1)
\left[
\psi(\tilde{\alpha}_{i,k})
- \psi\left(\sum_{j=1}^{K}\tilde{\alpha}_{i,j}\right)
\right],
\end{aligned}
$$

where $\Gamma(\cdot)$ is the Gamma function and $\psi(\cdot)$ is the digamma function. The final training objective is:

$$
\mathcal{L}
= \frac{1}{N}\sum_{i=1}^{N}
\left(
\mathcal{L}_{ECE}^{(i)}
+ \lambda_t \mathcal{L}_{KL}^{(i)}
\right),
$$

where $\lambda_t$ is an annealing coefficient that gradually increases during training. This strategy allows the model to first learn discriminative boundaries and then progressively suppress misleading evidence, reducing overconfidence on uncertain or misclassified samples.

## Uncertainty-aware Routing

The purpose of uncertainty-aware routing is to decide whether the small model's prediction is sufficiently reliable. For each sample $x_i$, the small model outputs a prediction $\hat{y}_i^{SM}$ and an uncertainty score $u_i$. \textsc{LogCo} compares $u_i$ with a threshold $\tau$:

$$
r_i =
\begin{cases}
0, & u_i \le \tau, \\
1, & u_i > \tau,
\end{cases}
$$

where $r_i=0$ means that the sample is directly handled by the small model, and $r_i=1$ means that the sample is escalated to the LLM. The threshold $\tau$ controls the trade-off between detection cost and robustness. A smaller threshold sends more samples to the LLM and may improve robustness at higher cost, while a larger threshold reduces LLM usage but relies more heavily on the small model.

This routing mechanism is central to \textsc{LogCo}. Unlike heuristic routing based on raw Softmax confidence, EDL-based uncertainty is explicitly derived from the amount of evidence collected by the model. In open-world scenarios, unseen templates and shifted log patterns often provide insufficient evidence, leading to higher uncertainty and making them more likely to be escalated for LLM reasoning.

In implementation, \textsc{LogCo} can determine $\tau$ using a validation set or a target LLM invocation budget. For example, if the system can afford to send at most $\rho$ proportion of logs to the LLM, $\tau$ can be selected as the $(1-\rho)$-quantile of validation uncertainty scores. This makes the framework adaptable to different operational budgets.

## Retrieval-augmented LLM Reasoning

For high-uncertainty samples, \textsc{LogCo} invokes an LLM reasoning module. Directly prompting the LLM with only the current log may lead to unstable or overly general answers. To ground the reasoning process in concrete historical evidence, \textsc{LogCo} uses retrieval-augmented generation (RAG).

The knowledge base stores historical logs, parsed templates, labels, uncertainty scores, and optional explanations. Each stored log is embedded into a vector representation and indexed in a vector database. Given an escalated sample $x_i$, \textsc{LogCo} retrieves the top-$M$ semantically similar historical cases:

$$
\mathcal{R}_i =
\mathrm{TopM}\left(
\mathrm{sim}(\mathbf{v}_i, \mathbf{v}_j),
\mathcal{D}_{KB}
\right),
$$

where $\mathbf{v}_i$ is the embedding of the current log, $\mathbf{v}_j$ is the embedding of a historical log, $\mathrm{sim}(\cdot,\cdot)$ denotes cosine similarity, and $\mathcal{D}_{KB}$ is the knowledge base.

The LLM prompt contains four types of information: the current raw log and parsed template, the small model's prediction and uncertainty score, the retrieved historical evidence, and the required output schema. The LLM is instructed to compare the current log with retrieved normal and abnormal cases, determine whether the log is normal or anomalous, and provide a concise explanation. The output is structured as:

$$
o_i^{LLM} =
(\hat{y}_i^{LLM}, c_i^{LLM}, e_i^{LLM}),
$$

where $\hat{y}_i^{LLM}$ is the predicted label, $c_i^{LLM}$ is the LLM confidence, and $e_i^{LLM}$ is the natural language explanation. This structured output improves interpretability and allows downstream modules to reuse high-confidence decisions.

To further reduce LLM cost, \textsc{LogCo} performs template-level deduplication before LLM invocation. If multiple high-uncertainty logs share the same normalized template, only one representative instance is sent to the LLM, and the resulting decision is mapped back to all corresponding samples. This strategy is especially useful in log data, where repeated templates are common.

## Feedback-driven Continual Adaptation

Open-world log anomaly detection requires the system to adapt to newly emerging patterns after deployment. \textsc{LogCo} therefore introduces a feedback-driven adaptation loop that transfers reliable LLM knowledge back to the system.

For each LLM-processed sample, if the LLM confidence exceeds a threshold $\eta$, its output is stored as a structured pseudo-labeled instance:

$$
\mathcal{D}_{fb}
= \left\{
(x_i, t_i, \hat{y}_i^{LLM}, e_i^{LLM})
\mid c_i^{LLM} \ge \eta
\right\}.
$$

These instances are inserted into the knowledge base and become retrievable evidence for future LLM reasoning. In addition, they can be used as supervision signals for periodic small-model updates. Let $\mathcal{D}_{old}$ denote a replay buffer of historical labeled samples and $\mathcal{D}_{fb}$ denote the newly collected feedback set. The small model can be updated on:

$$
\mathcal{D}_{adapt} = \mathcal{D}_{old} \cup \mathcal{D}_{fb}.
$$

This adaptation can be implemented through pseudo-label training, knowledge distillation, or replay-based fine-tuning. The purpose is not to continuously retrain the model after every log, but to periodically absorb stable knowledge from LLM reasoning. As the small model learns from newly confirmed patterns, future uncertainty on similar logs can decrease, thereby reducing subsequent LLM invocation. This closes the loop between efficient detection, semantic reasoning, and continual adaptation.

## Discussion

The design of \textsc{LogCo} separates routine detection from knowledge-intensive reasoning. The evidential small model provides efficient predictions and uncertainty estimates, while the LLM is used only when stronger semantic reasoning is needed. RAG grounds LLM decisions in historical evidence, and the feedback loop allows the system to accumulate knowledge over time. This design directly targets the open-world nature of log anomaly detection, where unseen templates, OOD samples, and concept drift are common after deployment.
