import json
import ast
from pathlib import Path
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from prompt_templates import (
    LOG_ANOMALY_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    RAG_SYSTEM_PROMPT_BINARY,
    RAG_USER_PROMPT_BINARY,
)
from vector_store import LogVectorStore
from config import (
    DATASET,
    DECISION_MODE,
    LLM_TYPE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


class HTTPChatLLM:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "thinking": {"type": "disabled"},
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return type(
            "HTTPResponse",
            (),
            {
                "content": content,
                "usage_metadata": {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
                "response_metadata": {"token_usage": usage},
            },
        )()


def get_llm():
    if LLM_TYPE == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("缺少 OPENAI_API_KEY，请先设置环境变量。")
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=OPENAI_MODEL,
                temperature=0.0,
            )
        except ModuleNotFoundError:
            return HTTPChatLLM(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=OPENAI_MODEL,
                temperature=0.0,
            )
    if LLM_TYPE == "ollama":
        from langchain_ollama import OllamaLLM
        print(f"🔧 使用 Ollama 模型: {OLLAMA_MODEL}，请确保 Ollama 服务已启动并加载该模型。")
        return OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )
    raise ValueError("LLM_TYPE 必须是 openai / ollama")

# 初始化
llm = get_llm()
_VECTOR_DB_CACHE = {}


def _get_vector_db(dataset: str) -> LogVectorStore:
    if dataset not in _VECTOR_DB_CACHE:
        store = LogVectorStore(dataset=dataset)
        doc_count = store.build_from_grouped_logs()
        print(f"✅ 已加载向量库: dataset={dataset}, 文档数={doc_count}")
        _VECTOR_DB_CACHE[dataset] = store
    return _VECTOR_DB_CACHE[dataset]


def _extract_content(result):
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return result.content
    return str(result)


def _safe_int(value):
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _estimate_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    text = "" if text is None else str(text)
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 粗略兜底：英文日志通常接近 4 chars/token，避免后端无 usage 时完全缺失。
        return max(1, (len(text) + 3) // 4) if text else 0


def _normalize_token_usage(raw_usage: dict | None):
    raw_usage = raw_usage or {}
    prompt_tokens = _safe_int(
        raw_usage.get("prompt_tokens")
        or raw_usage.get("input_tokens")
        or raw_usage.get("prompt_eval_count")
    )
    completion_tokens = _safe_int(
        raw_usage.get("completion_tokens")
        or raw_usage.get("output_tokens")
        or raw_usage.get("eval_count")
    )
    total_tokens = _safe_int(raw_usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_usage_source": "api",
    }


def _extract_token_usage(result, prompt: str, completion: str):
    usage = None
    usage_metadata = getattr(result, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        usage = usage_metadata

    response_metadata = getattr(result, "response_metadata", None)
    if not usage and isinstance(response_metadata, dict):
        usage = response_metadata.get("token_usage") or response_metadata.get("usage")

    if usage:
        normalized = _normalize_token_usage(usage)
        if normalized["total_tokens"] > 0:
            return normalized

    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "token_usage_source": "estimated",
    }


def _parse_result(text: str):
    # 解析结果，兼容 decision/status 和 confidence/level 两种表达方式
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}

    decision = parsed.get("decision", "").lower()
    status = parsed.get("status")
    if isinstance(status, str):
        status = status.strip().lower()
    else:
        status = None

    if not status:
        if decision == "anomaly":
            status = "anomaly"
        elif decision == "normal":
            status = "normal"
        elif decision == "uncertain":
            status = "uncertain"
        else:
            status = "unknown"
    elif status in {"0", "normal", "benign", "false", "ok"}:
        status = "normal"
    elif status in {"1", "anomaly", "anomalous", "true"}:
        status = "anomaly"

    confidence = parsed.get("confidence", None)
    if isinstance(confidence, (int, float)):
        if confidence >= 0.8:
            level = "high"
        elif confidence >= 0.5:
            level = "medium"
        else:
            level = "low"
    else:
        level = parsed.get("level", "unknown")

    return {
        "status": status,
        "level": level,
        "reason": parsed.get("rationale", parsed.get("reason", text)),
        "suggestion": parsed.get("recommended_action", parsed.get("suggestion", "N/A")),
    }


def detect(
    log: str,
    vector_db: LogVectorStore,
    small_model_score: str = "N/A",
    small_model_uncertainty: str = "N/A",
    verbose: bool = True,
    decision_mode: str = DECISION_MODE,
):
    if verbose:
        print("\n" + "="*60)
        print(f"日志：{log}")

    # 1. 检索相似日志
    docs = vector_db.search(log)
    context = "\n".join(
        [f"[R{i+1}] label={d.metadata.get('Label')}, text=\"{d.page_content}\"" for i, d in enumerate(docs)]
    )

    # 2. 构造提示词
    mode = str(decision_mode).strip().lower()
    if mode == "binary":
        system_prompt = RAG_SYSTEM_PROMPT_BINARY
        user_prompt_template = RAG_USER_PROMPT_BINARY
    else:
        system_prompt = RAG_SYSTEM_PROMPT
        user_prompt_template = RAG_USER_PROMPT

    user_prompt = user_prompt_template.format(
        retrieved_logs=context,
        target_log=log,
        small_model_score=small_model_score,
        small_model_uncertainty=small_model_uncertainty,
    )
    prompt = LOG_ANOMALY_PROMPT.format(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    # print(f"\n🧠 构造的提示词:\n{prompt}")
    # 3. LLM 调用
    result = llm.invoke(prompt)
    content = _extract_content(result)
    token_usage = _extract_token_usage(result, prompt, content)
    # print(f"\n💡 LLM 输出原始结果:\n{content}")
    # 4. 解析
    parsed = _parse_result(content)
    parsed["token_usage"] = token_usage
    if verbose:
        print(f"异常状态：{parsed['status']}")
        print(f"等级：{parsed['level']}")
        print(f"原因：{parsed['reason']}")
        print(f"建议：{parsed['suggestion']}")
        print(
            "Token消耗："
            f"prompt={token_usage['prompt_tokens']}, "
            f"completion={token_usage['completion_tokens']}, "
            f"total={token_usage['total_tokens']}, "
            f"source={token_usage['token_usage_source']}"
        )
    return parsed


def _templates_to_text(raw_value):
    if isinstance(raw_value, list):
        return " ".join(str(item) for item in raw_value)
    if pd.isna(raw_value):
        return ""

    text = str(raw_value).strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return " ".join(str(item) for item in parsed)
    except Exception:
        pass
    return text


def _resolve_uncertain_csv_path(dataset: str) -> Path:
    candidates = [
        Path("outputs") / dataset / "results" / "val_high_uncertain_samples.csv",
        Path("outputs") / dataset / "results" / "val_high_uncertain.csv",
        Path("outputs") / dataset / "result" / "hign_uncertain.csv",
        Path("outputs") / dataset / "results" / "hign_uncertain.csv",
        Path("outputs") / dataset / "result" / "high_uncertain.csv",
        Path("outputs") / dataset / "results" / "high_uncertain.csv",
        Path("outputs") / dataset / "results" / "high_uncertain_samples.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"未找到高不确定样本文件，请确认以下路径之一存在：{[str(p) for p in candidates]}"
    )


def _resolve_full_test_csv_path(dataset: str) -> Path:
    candidates = [
        Path("outputs") / dataset / "results" / "test_sm_predictions.csv",
        Path("outputs") / dataset / "results" / "sm_test_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"未找到全测试集预测文件，请确认以下路径之一存在：{[str(p) for p in candidates]}，"
        "请先运行 src/SM/Inference.py"
    )


def _run_llm_rag_on_df(
    df: pd.DataFrame,
    vector_db: LogVectorStore,
    dataset: str,
    decision_mode: str,
    output_prefix: str,
    input_csv: Path,
):
    if "Templates" not in df.columns:
        raise ValueError(f"{input_csv} 缺少 Templates 列")

    # 去重键：按标准化后的日志文本去重，避免重复调用 LLM
    df = df.copy()
    df["log_text"] = df["Templates"].apply(_templates_to_text)
    grouped = df.groupby("log_text", dropna=False)

    if "AnomalyScore" in df.columns:
        score_series = grouped["AnomalyScore"].mean().fillna("N/A")
    else:
        score_series = pd.Series("N/A", index=grouped.size().index)

    if "Uncertainty" in df.columns:
        unc_series = grouped["Uncertainty"].mean().fillna("N/A")
    else:
        unc_series = pd.Series("N/A", index=grouped.size().index)

    unique_df = pd.DataFrame(
        {
            "log_text": grouped.size().index,
            "dup_count": grouped.size().values,
            "small_model_score_mean": [score_series.loc[k] for k in grouped.size().index],
            "small_model_uncertainty_mean": [unc_series.loc[k] for k in grouped.size().index],
        }
    )

    results = []
    pbar = tqdm(unique_df.itertuples(index=False), total=len(unique_df), desc="LLM+RAG检测(去重后)")
    for row in pbar:
        parsed = detect(
            row.log_text,
            vector_db=vector_db,
            small_model_score=str(row.small_model_score_mean),
            small_model_uncertainty=str(row.small_model_uncertainty_mean),
            verbose=False,
            decision_mode=decision_mode,
        )
        results.append(
            {
                "log_text": row.log_text,
                "llm_status": parsed["status"],
                "llm_level": parsed["level"],
                "llm_reason": parsed["reason"],
                "llm_suggestion": parsed["suggestion"],
                "llm_prompt_tokens": parsed["token_usage"]["prompt_tokens"],
                "llm_completion_tokens": parsed["token_usage"]["completion_tokens"],
                "llm_total_tokens": parsed["token_usage"]["total_tokens"],
                "llm_token_usage_source": parsed["token_usage"]["token_usage_source"],
            }
        )

    result_df = pd.DataFrame(results)
    unique_result_df = unique_df.merge(result_df, on="log_text", how="left")
    # 映射回原始样本，保证后续指标按原样本统计
    df_out = df.merge(
        unique_result_df[
            [
                "log_text",
                "dup_count",
                "llm_status",
                "llm_level",
                "llm_reason",
                "llm_suggestion",
                "llm_prompt_tokens",
                "llm_completion_tokens",
                "llm_total_tokens",
                "llm_token_usage_source",
            ]
        ],
        on="log_text",
        how="left",
    )

    status_to_pred = {"normal": 0, "anomaly": 1}
    df_out["llm_pred"] = df_out["llm_status"].map(status_to_pred)

    out_dir = Path("outputs") / dataset / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{output_prefix}.csv"
    unique_out_csv = out_dir / f"{output_prefix}_unique.csv"

    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    unique_result_df.to_csv(unique_out_csv, index=False, encoding="utf-8")

    print(f"✅ LLM+RAG 检测完成，输入文件: {input_csv}")
    print(f"✅ 去重后请求数: {len(unique_result_df)} / 原始样本数: {len(df)}")
    print(f"✅ 结果已保存: {out_csv}")
    print(f"✅ 去重结果已保存: {unique_out_csv}")
    print(
        "✅ Token消耗(按实际LLM请求汇总): "
        f"prompt={int(unique_result_df['llm_prompt_tokens'].sum())}, "
        f"completion={int(unique_result_df['llm_completion_tokens'].sum())}, "
        f"total={int(unique_result_df['llm_total_tokens'].sum())}"
    )
    source_counts = unique_result_df["llm_token_usage_source"].value_counts().to_dict()
    print(f"✅ Token统计来源: {source_counts}")

    # 如包含标签则输出二次检测指标（按原始样本级）
    if "Label" in df_out.columns:
        metric_df = df_out.dropna(subset=["llm_pred"]).copy()
        if len(metric_df) > 0:
            y_true = metric_df["Label"].astype(int).to_numpy()
            y_pred = metric_df["llm_pred"].astype(int).to_numpy()
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            print(
                f"✅ 二次检测(样本级) Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, "
                f"覆盖率={len(metric_df)/len(df_out):.2%}"
            )
            if "AnomalyScore" in metric_df.columns:
                try:
                    auc = roc_auc_score(y_true, metric_df["AnomalyScore"].astype(float).to_numpy())
                    print(f"✅ 基于小模型分数的样本级AUC={auc:.4f}")
                except Exception:
                    pass

    return df_out, unique_result_df


def second_pass_for_high_uncertain(dataset: str = DATASET, decision_mode: str = DECISION_MODE):
    vector_db = _get_vector_db(dataset)
    input_csv = _resolve_uncertain_csv_path(dataset)
    df = pd.read_csv(input_csv)
    return _run_llm_rag_on_df(
        df=df,
        vector_db=vector_db,
        dataset=dataset,
        decision_mode=decision_mode,
        output_prefix="llm_second_pass_val_high_uncertain",
        input_csv=input_csv,
    )


def full_test_llm_rag(dataset: str = DATASET, decision_mode: str = DECISION_MODE):
    vector_db = _get_vector_db(dataset)
    input_csv = _resolve_full_test_csv_path(dataset)
    df = pd.read_csv(input_csv)
    return _run_llm_rag_on_df(
        df=df,
        vector_db=vector_db,
        dataset=dataset,
        decision_mode=decision_mode,
        output_prefix="llm_full_test_rag",
        input_csv=input_csv,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 LLM+RAG 日志异常检测实验")
    parser.add_argument("--dataset", default=DATASET, help="数据集名称，如 Spirit/BGL/HDFS/Thunderbird")
    parser.add_argument(
        "--scope",
        default="high_uncertain",
        choices=["high_uncertain", "full_test"],
        help="high_uncertain=仅高不确定样本, full_test=全测试集消融实验",
    )
    parser.add_argument(
        "--decision-mode",
        default=DECISION_MODE,
        choices=["triage", "binary"],
        help="triage=正常/异常/不确定, binary=仅正常/异常",
    )
    args = parser.parse_args()
    if args.scope == "full_test":
        full_test_llm_rag(dataset=args.dataset, decision_mode=args.decision_mode)
    else:
        second_pass_for_high_uncertain(dataset=args.dataset, decision_mode=args.decision_mode)
