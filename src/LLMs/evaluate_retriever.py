import argparse
import ast
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    DATASET,
    TOP_K,
    UNCERTAINTY_SPLIT,
)
from vector_store import (
    DEFAULT_CONTRASTIVE_FETCH_MULTIPLIER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT_MODE,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_USE_TRAIN_ONLY,
    DEFAULT_VECTOR_SOURCE,
    LogVectorStore,
)


def _templates_to_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return "\n".join(str(item).strip() for item in parsed if str(item).strip())
    except Exception:
        pass
    return text


def _label_to_int(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"0", "normal", "benign", "false", "ok", "-"}:
        return 0
    if text in {"1", "anomaly", "anomalous", "true"}:
        return 1
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _default_input_csv(dataset: str, split: str) -> Path:
    return Path("outputs") / dataset / "results" / f"{split}_high_uncertain_samples.csv"


def _doc_summary(doc, score):
    return {
        "score": float(score),
        "label": _label_to_int(doc.metadata.get("Label")),
        "dup_count": doc.metadata.get("dup_count"),
        "text": str(doc.page_content).replace("\n", " | ")[:500],
    }


def _best_score(scored_docs):
    if not scored_docs:
        return None
    return float(scored_docs[0][1])


def _resolve_label_col(df: pd.DataFrame, label_col: str | None) -> str | None:
    if label_col:
        if label_col not in df.columns:
            raise ValueError(f"指定的标签列不存在: {label_col}")
        return label_col
    if "Label" in df.columns:
        return "Label"
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    for col in unnamed_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) and values.isin([0, 1]).mean() > 0.95:
            return col
    return None


def _evaluate_row(store: LogVectorStore, row: dict, label_col: str | None, row_index: int, top_k: int, fetch_multiplier: int):
    query_text = _templates_to_text(row.get("Templates", ""))
    true_label = _label_to_int(row.get(label_col)) if label_col else None
    result = store.contrastive_search_with_scores(
        query_text,
        top_k=top_k,
        fetch_multiplier=fetch_multiplier,
    )

    normal_docs = result["normal"]
    anomaly_docs = result["anomaly"]
    normal_best = _best_score(normal_docs)
    anomaly_best = _best_score(anomaly_docs)

    if normal_best is None and anomaly_best is None:
        nearest_side = None
    elif anomaly_best is None:
        nearest_side = 0
    elif normal_best is None:
        nearest_side = 1
    else:
        nearest_side = 1 if anomaly_best < normal_best else 0

    if true_label == 1:
        same_best = anomaly_best
        other_best = normal_best
    elif true_label == 0:
        same_best = normal_best
        other_best = anomaly_best
    else:
        same_best = None
        other_best = None

    same_wins = None
    margin = None
    if same_best is not None and other_best is not None:
        same_wins = same_best <= other_best
        margin = other_best - same_best

    return {
        "row_index": int(row_index),
        "TestIndex": row.get("TestIndex"),
        "Label": true_label,
        "Pred": row.get("Pred"),
        "AnomalyScore": row.get("AnomalyScore"),
        "Uncertainty": row.get("Uncertainty"),
        "query_text": query_text.replace("\n", " | ")[:1000],
        "normal_count": len(normal_docs),
        "anomaly_count": len(anomaly_docs),
        "normal_best_score": normal_best,
        "anomaly_best_score": anomaly_best,
        "nearest_side_pred": nearest_side,
        "same_label_best_score": same_best,
        "other_label_best_score": other_best,
        "same_label_wins": same_wins,
        "score_margin_other_minus_same": margin,
        "normal_refs": json.dumps([_doc_summary(doc, score) for doc, score in normal_docs], ensure_ascii=False),
        "anomaly_refs": json.dumps([_doc_summary(doc, score) for doc, score in anomaly_docs], ensure_ascii=False),
    }


def _summarize(df: pd.DataFrame, top_k: int) -> dict:
    labeled = df[df["Label"].isin([0, 1])].copy()
    summary = {
        "rows": int(len(df)),
        "labeled_rows": int(len(labeled)),
        "top_k_per_class": int(top_k),
        "normal_group_full_rate": float((df["normal_count"] >= top_k).mean()) if len(df) else 0.0,
        "anomaly_group_full_rate": float((df["anomaly_count"] >= top_k).mean()) if len(df) else 0.0,
        "avg_normal_count": float(df["normal_count"].mean()) if len(df) else 0.0,
        "avg_anomaly_count": float(df["anomaly_count"].mean()) if len(df) else 0.0,
    }

    if len(labeled):
        valid_pred = labeled[labeled["nearest_side_pred"].isin([0, 1])]
        valid_margin = labeled[labeled["score_margin_other_minus_same"].notna()]
        summary.update(
            {
                "nearest_side_accuracy": float((valid_pred["nearest_side_pred"] == valid_pred["Label"]).mean())
                if len(valid_pred)
                else None,
                "same_label_win_rate": float(valid_margin["same_label_wins"].mean()) if len(valid_margin) else None,
                "avg_score_margin_other_minus_same": float(valid_margin["score_margin_other_minus_same"].mean())
                if len(valid_margin)
                else None,
                "label_distribution": {
                    str(k): int(v) for k, v in labeled["Label"].value_counts(dropna=False).to_dict().items()
                },
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="评估 RAG 历史日志序列检索器质量，不调用 LLM")
    parser.add_argument("--dataset", default=DATASET, help="数据集名称，如 BGL/Spirit")
    parser.add_argument("--input-csv", default=None, help="待评估样本 CSV，默认使用对应 split 的 high_uncertain 文件")
    parser.add_argument("--split", default=UNCERTAINTY_SPLIT, help="默认输入文件使用的 split，例如 val/test")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="每类返回的正常/异常参考数量")
    parser.add_argument(
        "--fetch-multiplier",
        type=int,
        default=DEFAULT_CONTRASTIVE_FETCH_MULTIPLIER,
        help="对比检索候选池放大倍数",
    )
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条，0 表示全量")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认 outputs/<DATASET>/results")
    parser.add_argument("--label-col", default=None, help="标签列名；默认自动使用 Label 或二值 Unnamed 列")
    parser.add_argument("--split-mode", default=DEFAULT_SPLIT_MODE, choices=["ordered", "random"])
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--vector-source", default=DEFAULT_VECTOR_SOURCE, choices=["grouped", "structured"])
    parser.add_argument("--use-all-data", action="store_true", help="使用全量向量库，默认仅训练段")
    args = parser.parse_args()

    input_csv = Path(args.input_csv) if args.input_csv else _default_input_csv(args.dataset, args.split)
    if not input_csv.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_csv}")

    df = pd.read_csv(input_csv)
    if "Templates" not in df.columns:
        raise ValueError(f"{input_csv} 缺少 Templates 列")
    label_col = _resolve_label_col(df, args.label_col)
    if label_col:
        print(f"✅ 使用标签列: {label_col}")
    else:
        print("⚠️ 未找到标签列，仅统计检索覆盖率，不计算同标签胜率")
    if args.limit > 0:
        df = df.head(args.limit).copy()

    store = LogVectorStore(
        dataset=args.dataset,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        use_train_only=not args.use_all_data and DEFAULT_USE_TRAIN_ONLY,
        vector_source=args.vector_source,
    )
    doc_count = store.build_from_grouped_logs()
    print(f"✅ 已加载向量库: {store.index_dir}, 文档数={doc_count}")
    print(f"✅ 评估输入: {input_csv}, 样本数={len(df)}")

    rows = []
    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="评估检索器"):
        rows.append(_evaluate_row(store, row.to_dict(), label_col, row_index, args.top_k, args.fetch_multiplier))

    detail_df = pd.DataFrame(rows)
    summary = _summarize(detail_df, args.top_k)

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / args.dataset / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"retriever_quality_{input_csv.stem}_top{args.top_k}"
    detail_path = out_dir / f"{stem}.csv"
    summary_path = out_dir / f"{stem}.json"
    failure_path = out_dir / f"{stem}_failures.csv"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    failures = detail_df[
        detail_df["Label"].isin([0, 1])
        & detail_df["nearest_side_pred"].isin([0, 1])
        & (detail_df["nearest_side_pred"] != detail_df["Label"])
    ].copy()
    failures.to_csv(failure_path, index=False, encoding="utf-8")

    print("\n========== Retriever Quality ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\n✅ 明细: {detail_path}")
    print(f"✅ 摘要: {summary_path}")
    print(f"✅ 失败样本: {failure_path}")


if __name__ == "__main__":
    main()
