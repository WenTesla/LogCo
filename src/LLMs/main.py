import json
import ast
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from llm_client import get_llm
from prompt_templates import (
    LOG_ANOMALY_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    RAG_SYSTEM_PROMPT_BINARY,
    RAG_USER_PROMPT_BINARY,
)
from vector_store import LogVectorStore
from config import DATASET, DECISION_MODE

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


def _parse_result(text: str):
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}

    decision = parsed.get("decision", "").lower()
    status = parsed.get("status")
    if not status:
        if decision == "anomaly":
            status = "anomaly"
        elif decision == "normal":
            status = "normal"
        elif decision == "uncertain":
            status = "uncertain"
        else:
            status = "unknown"

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

    # 3. LLM 调用
    result = llm.invoke(prompt)

    # 4. 解析
    parsed = _parse_result(_extract_content(result))
    if verbose:
        print(f"异常状态：{parsed['status']}")
        print(f"等级：{parsed['level']}")
        print(f"原因：{parsed['reason']}")
        print(f"建议：{parsed['suggestion']}")
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


def second_pass_for_high_uncertain(dataset: str = DATASET, decision_mode: str = DECISION_MODE):
    vector_db = _get_vector_db(dataset)
    input_csv = _resolve_uncertain_csv_path(dataset)
    df = pd.read_csv(input_csv)
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
    pbar = tqdm(unique_df.itertuples(index=False), total=len(unique_df), desc="LLM二次检测(去重后)")
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
            }
        )

    unique_result_df = unique_df.merge(pd.DataFrame(results), on="log_text", how="left")
    # 映射回原始样本，保证后续指标按原样本统计
    df_out = df.merge(
        unique_result_df[["log_text", "dup_count", "llm_status", "llm_level", "llm_reason", "llm_suggestion"]],
        on="log_text",
        how="left",
    )

    status_to_pred = {"normal": 0, "anomaly": 1}
    df_out["llm_pred"] = df_out["llm_status"].map(status_to_pred)

    out_dir = Path("outputs") / dataset / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "llm_second_pass_high_uncertain.csv"
    unique_out_csv = out_dir / "llm_second_pass_high_uncertain_unique.csv"

    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    unique_result_df.to_csv(unique_out_csv, index=False, encoding="utf-8")

    print(f"✅ 二次检测完成，仅处理文件: {input_csv}")
    print(f"✅ 去重后请求数: {len(unique_result_df)} / 原始样本数: {len(df)}")
    print(f"✅ 结果已保存: {out_csv}")
    print(f"✅ 去重结果已保存: {unique_out_csv}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="仅对高不确定样本做 LLM 二次检测")
    parser.add_argument("--dataset", default=DATASET, help="数据集名称，如 Spirit/BGL/HDFS/Thunderbird")
    parser.add_argument(
        "--decision-mode",
        default=DECISION_MODE,
        choices=["triage", "binary"],
        help="triage=正常/异常/不确定, binary=仅正常/异常",
    )
    args = parser.parse_args()
    second_pass_for_high_uncertain(dataset=args.dataset, decision_mode=args.decision_mode)
