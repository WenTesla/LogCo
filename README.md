# LogCo: Cascaded Log Anomaly Detection (SM + LLM-RAG)

本仓库实现两阶段日志异常检测：

1. 小模型（SM/EDL）先做异常判断与不确定性估计；
2. 高不确定样本进入 LLM + RAG 做二次判别；
3. 支持在高不确定子集与全测试集两种范围做级联评估。

## Project Structure

```text
LogCo/
  Preprocess.py
  pyproject.toml
  uv.lock
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
    evaluate_cascade.py
```

## Environment Setup (uv)

推荐使用 `uv`：

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync
uv sync --group sm
```

可选：HuggingFace 镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Unified Config (SM + LLM)

SM 与 LLM 支持统一环境变量：

```bash
export DATASET=Spirit                 # Spirit/BGL/HDFS/Thunderbird
export SPLIT_MODE=ordered             # ordered/random
export TRAIN_RATIO=0.7
export RANDOM_SEED=42
```

LLM 额外配置：

```bash
export LLM_TYPE=openai                # mock/openai/ollama
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=deepseek-v4-flash
```

## End-to-End Pipeline

1. 数据预处理

```bash
python Preprocess.py
```

2. 训练 SM

```bash
python src/SM/Train.py
```

3. SM 推理与不确定样本导出

```bash
python src/SM/Inference.py
python src/SM/UncertaintyAnalysis.py
```

`Inference.py` 产物（`outputs/<DATASET>/results/`）：
- `sm_test_predictions.csv`（全测试集，含 `TestIndex`）
- `high_uncertain_samples.csv`
- `low_uncertain_samples.csv`
- `inference_metrics_history.json`

4. 构建 RAG 向量库（支持与 SM 同样切分）

```bash
python src/LLMs/vector_store.py \
  --dataset Spirit \
  --split-mode ordered \
  --train-ratio 0.7 \
  --random-seed 42 \
  --force-rebuild
```

说明：
- 默认仅用训练部分建库（避免泄漏）；
- `--use-all-data` 可改为全量建库；
- 索引目录会包含切分参数，例如 `faiss_bge3_ordered_0p7_seed42`。

5. 高不确定样本二次检测（去重后调用 LLM）

```bash
python src/LLMs/main.py --dataset Spirit
```

产物（`outputs/<DATASET>/results/`）：
- `llm_second_pass_high_uncertain_unique.csv`（去重粒度）
- `llm_second_pass_high_uncertain.csv`（映射回样本粒度）

6. 级联评估

全测试集级联评估（推荐）：

```bash
python src/evaluate_cascade.py --dataset Spirit --scope full_test
```

仅高不确定子集评估：

```bash
python src/evaluate_cascade.py --dataset Spirit --scope high_uncertain
```

可选参数：
- `--uncertain-policy`：`fallback_sm` / `as_anomaly` / `as_normal` / `drop`

评估报告保存为：
- `outputs/<DATASET>/results/cascade_eval_report_full_test.json`
- `outputs/<DATASET>/results/cascade_eval_report_high_uncertain.json`

## Notes

- 默认 LLM 后端为 `openai`。
- 若需离线验证可设 `LLM_TYPE=mock`。
- 向量库默认 embedding 为 `BAAI/bge-small-en-v1.5`，可通过 `BGE3_MODEL_NAME` 覆盖。
