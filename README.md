# LogCo: Cascaded Log Anomaly Detection (Small Model + LLM-RAG)

本仓库实现两阶段日志异常检测：

1. 小模型（EDL）进行快速初筛与不确定性估计；
2. 高不确定性样本进入 LLM + RAG 进行二次判别。



## 开启代理


export HF_ENDPOINT=https://hf-mirror.com


## Project Structure（当前仓库）

```
LogCo/
  Preprocess.py
  src/
    SM/
      config.py
      LogDataset.py
      Model.py
      Train.py
      Inference.py
      UncertaintyAnalysis.py
    LLMs/
      config.py
      llm_client.py
      prompt_templates.py
      vector_store.py
      main.py
  requirements.txt
  script/
    Down.sh
  README.md
```

## Quick Start

1. 安装依赖：

```bash
pip install -r requirements.txt
```

或使用 `uv`（推荐，创建独立虚拟环境）：

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install requests openai langchain-openai
# 若需运行小模型（SM），再安装：
uv pip install torch==2.11.0 torchvision==0.26.0 transformers pandas numpy tqdm scikit-learn matplotlib
```

2. 运行 LLM 演示（默认 `DeepSeek V4 Flash`）：

```bash
export OPENAI_API_KEY=你的DeepSeek密钥
export OPENAI_MODEL=deepseek-v4-flash
python src/LLMs/main.py
```

3. 运行小模型流程（先预处理，再训练/推理）：

```bash
python Preprocess.py
python src/SM/Train.py
python src/SM/Inference.py
python src/SM/UncertaintyAnalysis.py
```

## Method Summary

For each log sample `x`:

- Small model outputs anomaly score `p_s(x)` and uncertainty `u_s(x)`.
- If `u_s(x) <= tau_u`, use stage-1 result directly.
- If `u_s(x) > tau_u`, retrieve top-k context from knowledge base and ask the LLM judge for second-pass decision.

This design keeps stage-1 fast while reserving expensive reasoning for difficult cases.

## Notes

- LLM 默认后端为 `openai`，默认模型为 `deepseek-v4-flash`。
- 若要离线测试可设置 `LLM_TYPE=mock`。
- 若使用 Ollama：设置 `LLM_TYPE=ollama` 及对应环境变量。
