import json
import ast
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from prompt_templates import (
    LOG_ANOMALY_PROMPT,
    RAG_SYSTEM_PROMPT_BINARY,
    RAG_SYSTEM_PROMPT_BINARY_RULE_ONLY,
    RAG_SYSTEM_PROMPT_BINARY_WITH_RULES,
    RAG_USER_PROMPT_BINARY,
    RAG_USER_PROMPT_BINARY_RULE_ONLY,
    RAG_USER_PROMPT_BINARY_WITH_RULES,
)
from rule_store import RuleStore, format_rules_for_prompt
from vector_store import LogVectorStore
from config import (
    DATASET,
    DECISION_MODE,
    HIGH_UNCERTAIN_CSV_PATH,
    LLM_TYPE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    RAG_CONTEXT_MODE,
    TOP_K,
    UNCERTAINTY_SPLIT,
)


def get_llm():
    if LLM_TYPE == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("缺少 OPENAI_API_KEY，请先设置环境变量。")
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        def invoke(prompt: str):
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            r = response
            return type(
                "_Response",
                (),
                {
                    "content": r.choices[0].message.content,
                },
            )()

        return type("OpenAILLM", (), {"invoke": invoke})()
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
_RULE_STORE_CACHE = {}
VALID_RAG_CONTEXT_MODES = {"history_only", "rule_only", "hybrid"}


def _normalize_rag_context_mode(mode: str = RAG_CONTEXT_MODE) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_RAG_CONTEXT_MODES:
        raise ValueError(
            "RAG_CONTEXT_MODE must be one of "
            f"{sorted(VALID_RAG_CONTEXT_MODES)}, got {mode!r}"
        )
    return normalized


def _uses_history(mode: str) -> bool:
    return mode in {"history_only", "hybrid"}


def _uses_rules(mode: str) -> bool:
    return mode in {"rule_only", "hybrid"}


def _get_vector_db(dataset: str) -> LogVectorStore:
    if dataset not in _VECTOR_DB_CACHE:
        store = LogVectorStore(dataset=dataset)
        doc_count = store.build_from_grouped_logs()
        print(f"✅ 已加载向量库: dataset={dataset}, 文档数={doc_count}")
        _VECTOR_DB_CACHE[dataset] = store
    return _VECTOR_DB_CACHE[dataset]


def _get_rule_store(dataset: str) -> RuleStore:
    if dataset not in _RULE_STORE_CACHE:
        store = RuleStore(dataset=dataset)
        print(f"✅ 已加载规则库: dataset={dataset}, 规则数={len(store.rules)}")
        _RULE_STORE_CACHE[dataset] = store
    return _RULE_STORE_CACHE[dataset]


def _extract_content(result):
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return result.content
    return str(result)


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


def _label_to_int(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"0", "normal", "benign", "false", "ok"}:
        return 0
    if text in {"1", "anomaly", "anomalous", "true"}:
        return 1
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _retrieval_diagnostics(docs):
    labels = [_label_to_int(d.metadata.get("Label")) for d in docs]
    normal_count = sum(1 for label in labels if label == 0)
    anomaly_count = sum(1 for label in labels if label == 1)
    unknown_count = sum(1 for label in labels if label is None)
    refs = []
    for i, doc in enumerate(docs, start=1):
        text = str(doc.page_content).replace("\n", " | ")
        refs.append(
            {
                "ref": f"R{i}",
                "label": labels[i - 1],
                "dup_count": doc.metadata.get("dup_count"),
                "text": text[:500],
            }
        )
    return {
        "retrieved_labels": labels,
        "retrieved_normal_count": normal_count,
        "retrieved_anomaly_count": anomaly_count,
        "retrieved_unknown_count": unknown_count,
        "retrieved_refs": refs,
        "retrieved_normal_refs": [ref for ref in refs if ref["label"] == 0],
        "retrieved_anomaly_refs": [ref for ref in refs if ref["label"] == 1],
    }


def _format_doc_ref(doc, ref: str) -> str:
    text = str(doc.page_content).replace("\n", " | ")
    return f"[{ref}] label={doc.metadata.get('Label')}, dup_count={doc.metadata.get('dup_count')}, text=\"{text}\""


def _format_contrastive_context(normal_docs, anomaly_docs):
    normal_context = "\n".join(
        _format_doc_ref(doc, f"N{i}") for i, doc in enumerate(normal_docs, start=1)
    )
    anomaly_context = "\n".join(
        _format_doc_ref(doc, f"A{i}") for i, doc in enumerate(anomaly_docs, start=1)
    )
    return "\n\n".join(
        [
            "Similar NORMAL log sequences:",
            normal_context or "None retrieved.",
            "Similar ANOMALY log sequences:",
            anomaly_context or "None retrieved.",
        ]
    )


def _apply_precision_guard(parsed: dict, diagnostics: dict, decision_mode: str) -> dict:
    if parsed.get("status") != "anomaly":
        return parsed

    normal_count = diagnostics["retrieved_normal_count"]
    anomaly_count = diagnostics["retrieved_anomaly_count"]
    if normal_count <= anomaly_count:
        return parsed

    guarded = parsed.copy()
    if str(decision_mode).strip().lower() == "binary":
        guarded["status"] = "normal"
    else:
        guarded["status"] = "uncertain"
    guarded["level"] = "low"
    guarded["reason"] = (
        "Precision guard applied: retrieved references are mostly normal, "
        "so the anomaly decision is not accepted without stronger evidence. "
        f"Original reason: {parsed.get('reason', '')}"
    )
    return guarded


def detect(
    log: str,
    vector_db: LogVectorStore | None = None,
    rule_store: RuleStore | None = None,
    small_model_score: str = "N/A",
    small_model_uncertainty: str = "N/A",
    verbose: bool = True,
    decision_mode: str = DECISION_MODE,
    rag_context_mode: str = RAG_CONTEXT_MODE,
):
    rag_context_mode = _normalize_rag_context_mode(rag_context_mode)
    if verbose:
        print("\n" + "="*60)
        print(f"日志：{log}")

    # 1. 按消融模式检索历史日志和/或规则
    if _uses_history(rag_context_mode):
        if vector_db is None:
            raise ValueError(f"{rag_context_mode} mode requires vector_db")
        retrieval_result = vector_db.contrastive_search(log, top_k=TOP_K)
        normal_docs = retrieval_result["normal"]
        anomaly_docs = retrieval_result["anomaly"]
        docs = retrieval_result["docs"]
        context = _format_contrastive_context(normal_docs, anomaly_docs)
    else:
        docs = []
        normal_docs = []
        anomaly_docs = []
        context = ""
    diagnostics = _retrieval_diagnostics(docs)
    normal_diagnostics = _retrieval_diagnostics(normal_docs)
    anomaly_diagnostics = _retrieval_diagnostics(anomaly_docs)
    diagnostics["retrieved_normal_refs"] = normal_diagnostics["retrieved_refs"]
    diagnostics["retrieved_anomaly_refs"] = anomaly_diagnostics["retrieved_refs"]
    diagnostics["contrastive_top_k_per_class"] = TOP_K
    if _uses_rules(rag_context_mode):
        rule_store = rule_store or _get_rule_store(DATASET)
        retrieved_rules = rule_store.search(log)
        rule_context = format_rules_for_prompt(retrieved_rules)
    else:
        retrieved_rules = []
        rule_context = ""

    # 2. 构造提示词
    mode = str(decision_mode).strip().lower()
    if mode == "binary":
        if rag_context_mode == "history_only":
            system_prompt = RAG_SYSTEM_PROMPT_BINARY
            user_prompt_template = RAG_USER_PROMPT_BINARY
        elif rag_context_mode == "rule_only":
            system_prompt = RAG_SYSTEM_PROMPT_BINARY_RULE_ONLY
            user_prompt_template = RAG_USER_PROMPT_BINARY_RULE_ONLY
        else:
            system_prompt = RAG_SYSTEM_PROMPT_BINARY_WITH_RULES
            user_prompt_template = RAG_USER_PROMPT_BINARY_WITH_RULES
    else:
        system_prompt = ""
        user_prompt_template = ""

    user_prompt = user_prompt_template.format(
        retrieved_logs=context,
        retrieved_rules=rule_context,
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
    # print(f"\n💡 LLM 输出原始结果:\n{content}")
    # 4. 解析
    parsed = _parse_result(content)
    if _uses_history(rag_context_mode):
        parsed = _apply_precision_guard(parsed, diagnostics, decision_mode=decision_mode)
    parsed["llm_raw_output"] = content
    parsed["retrieval"] = diagnostics
    parsed["rules"] = retrieved_rules
    parsed["rag_context_mode"] = rag_context_mode
    if verbose:
        print(f"异常状态：{parsed['status']}")
        print(f"等级：{parsed['level']}")
        print(f"原因：{parsed['reason']}")
        print(f"建议：{parsed['suggestion']}")
        print(
            "检索标签："
            f"normal={diagnostics['retrieved_normal_count']}, "
            f"anomaly={diagnostics['retrieved_anomaly_count']}, "
            f"unknown={diagnostics['retrieved_unknown_count']}"
        )
        print(f"命中规则：{[rule['rule_id'] for rule in retrieved_rules]}")
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

# ===================== 主流程 =====================

def _resolve_uncertain_csv_path(dataset: str) -> Path:
    if HIGH_UNCERTAIN_CSV_PATH:
        path = Path(
            HIGH_UNCERTAIN_CSV_PATH.format(
                dataset=dataset,
                split=UNCERTAINTY_SPLIT,
            )
        )
    else:
        path = (
            Path("outputs")
            / dataset
            / "results"
            / f"{UNCERTAINTY_SPLIT}_high_uncertain_samples.csv"
        )

    if path.exists():
        return path
    raise FileNotFoundError(
        f"未找到高不确定样本文件: {path}。"
        "请先运行 src/SM/UncertaintyAnalysis.py，或通过 HIGH_UNCERTAIN_CSV_PATH 注入路径。"
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
    vector_db: LogVectorStore | None,
    dataset: str,
    decision_mode: str,
    output_prefix: str,
    input_csv: Path,
    rag_context_mode: str = RAG_CONTEXT_MODE,
):
    rag_context_mode = _normalize_rag_context_mode(rag_context_mode)
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

    rule_store = _get_rule_store(dataset) if _uses_rules(rag_context_mode) else None
    results = []
    pbar = tqdm(unique_df.itertuples(index=False), total=len(unique_df), desc="LLM+RAG检测(去重后)")
    for row in pbar:
        parsed = detect(
            row.log_text,
            vector_db=vector_db,
            rule_store=rule_store,
            small_model_score=str(row.small_model_score_mean),
            small_model_uncertainty=str(row.small_model_uncertainty_mean),
            verbose=False,
            decision_mode=decision_mode,
            rag_context_mode=rag_context_mode,
        )
        results.append(
            {
                "log_text": row.log_text,
                "rag_context_mode": parsed["rag_context_mode"],
                "llm_status": parsed["status"],
                "llm_level": parsed["level"],
                "llm_reason": parsed["reason"],
                "llm_suggestion": parsed["suggestion"],
                "llm_raw_output": parsed["llm_raw_output"],
                "retrieved_labels": json.dumps(parsed["retrieval"]["retrieved_labels"], ensure_ascii=False),
                "retrieved_normal_count": parsed["retrieval"]["retrieved_normal_count"],
                "retrieved_anomaly_count": parsed["retrieval"]["retrieved_anomaly_count"],
                "retrieved_unknown_count": parsed["retrieval"]["retrieved_unknown_count"],
                "retrieved_refs": json.dumps(parsed["retrieval"]["retrieved_refs"], ensure_ascii=False),
                "retrieved_normal_refs": json.dumps(parsed["retrieval"]["retrieved_normal_refs"], ensure_ascii=False),
                "retrieved_anomaly_refs": json.dumps(parsed["retrieval"]["retrieved_anomaly_refs"], ensure_ascii=False),
                "retrieved_rule_ids": json.dumps(
                    [rule["rule_id"] for rule in parsed["rules"]],
                    ensure_ascii=False,
                ),
                "retrieved_rules": json.dumps(parsed["rules"], ensure_ascii=False),
            }
        )

    result_df = pd.DataFrame(results)
    unique_result_df = unique_df.merge(result_df, on="log_text", how="left")
    status_to_pred = {"normal": 0, "anomaly": 1}
    unique_result_df["llm_pred"] = unique_result_df["llm_status"].map(status_to_pred)
    # 映射回原始样本，保证后续指标按原样本统计
    df_out = df.merge(
        unique_result_df[
            [
                "log_text",
                "dup_count",
                "rag_context_mode",
                "llm_status",
                "llm_level",
                "llm_reason",
                "llm_suggestion",
                "llm_raw_output",
                "llm_pred",
                "retrieved_labels",
                "retrieved_normal_count",
                "retrieved_anomaly_count",
                "retrieved_unknown_count",
                "retrieved_refs",
                "retrieved_normal_refs",
                "retrieved_anomaly_refs",
                "retrieved_rule_ids",
                "retrieved_rules",
            ]
        ],
        on="log_text",
        how="left",
    )

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
    # 如包含标签则输出二次检测指标（按原始样本级）
    if "Label" in df_out.columns:
        metric_df = df_out.dropna(subset=["llm_pred"]).copy()
        if len(metric_df) > 0:
            y_true = metric_df["Label"].astype(int).to_numpy()
            y_pred = metric_df["llm_pred"].astype(int).to_numpy()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            print(
                f"✅ 二次检测(样本级) TP={tp} FP={fp} TN={tn} FN={fn} | "
                f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} | "
                f"覆盖率={len(metric_df)/len(df_out):.2%}"
            )
            if "AnomalyScore" in metric_df.columns:
                try:
                    auc = roc_auc_score(y_true, metric_df["AnomalyScore"].astype(float).to_numpy())
                    print(f"✅ 基于小模型分数的样本级AUC={auc:.4f}")
                except Exception:
                    pass

    return df_out, unique_result_df


def second_pass_for_high_uncertain(
    dataset: str = DATASET,
    decision_mode: str = DECISION_MODE,
    rag_context_mode: str = RAG_CONTEXT_MODE,
):
    rag_context_mode = _normalize_rag_context_mode(rag_context_mode)
    vector_db = _get_vector_db(dataset) if _uses_history(rag_context_mode) else None
    input_csv = _resolve_uncertain_csv_path(dataset)
    df = pd.read_csv(input_csv)
    print(
        f"✅ 发现高不确定样本文件: {input_csv}，共 {len(df)} 条样本，"
        f"RAG_CONTEXT_MODE={rag_context_mode}, 历史日志检索=contrastive。"
    )
    return _run_llm_rag_on_df(
        df=df,
        vector_db=vector_db,
        dataset=dataset,
        decision_mode=decision_mode,
        output_prefix=f"llm_second_pass_{UNCERTAINTY_SPLIT}_high_uncertain_{rag_context_mode}_contrastive",
        input_csv=input_csv,
        rag_context_mode=rag_context_mode,
    )


def full_test_llm_rag(
    dataset: str = DATASET,
    decision_mode: str = DECISION_MODE,
    rag_context_mode: str = RAG_CONTEXT_MODE,
):
    rag_context_mode = _normalize_rag_context_mode(rag_context_mode)
    vector_db = _get_vector_db(dataset) if _uses_history(rag_context_mode) else None
    input_csv = _resolve_full_test_csv_path(dataset)
    df = pd.read_csv(input_csv)
    return _run_llm_rag_on_df(
        df=df,
        vector_db=vector_db,
        dataset=dataset,
        decision_mode=decision_mode,
        output_prefix=f"llm_full_test_rag_{rag_context_mode}_contrastive",
        input_csv=input_csv,
        rag_context_mode=rag_context_mode,
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
        "--rag-context-mode",
        default=RAG_CONTEXT_MODE,
        choices=sorted(VALID_RAG_CONTEXT_MODES),
        help="history_only=仅历史日志RAG, rule_only=仅规则RAG, hybrid=规则+历史日志RAG",
    )
    args = parser.parse_args()
    if args.scope == "full_test":
        full_test_llm_rag(
            dataset=args.dataset,
            rag_context_mode=args.rag_context_mode,
        )
    else:
        second_pass_for_high_uncertain(
            dataset=args.dataset,
            rag_context_mode=args.rag_context_mode,
        )
