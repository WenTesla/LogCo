import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def _to_binary(series: pd.Series) -> pd.Series:
    mapped = series.copy()
    if mapped.dtype == object:
        mapped = mapped.astype(str).str.strip().str.lower().map(
            {"0": 0, "1": 1, "normal": 0, "anomaly": 1}
        )
    return pd.to_numeric(mapped, errors="coerce")


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series, score: Optional[pd.Series] = None) -> Dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    yt = y_true[mask].astype(int).to_numpy()
    yp = y_pred[mask].astype(int).to_numpy()

    if len(yt) == 0:
        return {
            "coverage": 0.0,
            "support": 0,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "accuracy": np.nan,
            "auc": np.nan,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    acc = accuracy_score(yt, yp)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    auc = np.nan
    if score is not None:
        sm = pd.to_numeric(score[mask], errors="coerce")
        valid = sm.notna()
        if valid.any() and len(np.unique(yt[valid.to_numpy()])) > 1:
            auc = float(roc_auc_score(yt[valid.to_numpy()], sm[valid].to_numpy()))

    return {
        "coverage": float(mask.mean()),
        "support": int(mask.sum()),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _resolve_results_file(dataset: str, filename: str, explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = Path("outputs") / dataset / "results" / filename
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    return path


def _build_llm_pred(df: pd.DataFrame, uncertain_policy: str, fallback_sm_pred: pd.Series) -> Tuple[pd.Series, str]:
    llm_pred = _to_binary(df["llm_pred"]) if "llm_pred" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    if "llm_status" in df.columns:
        llm_status_pred = df["llm_status"].astype(str).str.strip().str.lower().map({"normal": 0, "anomaly": 1})
        llm_pred = llm_pred.where(~llm_pred.isna(), llm_status_pred)

    if uncertain_policy == "fallback_sm":
        llm_pred = llm_pred.where(~llm_pred.isna(), fallback_sm_pred)
    elif uncertain_policy == "as_anomaly":
        llm_pred = llm_pred.where(~llm_pred.isna(), 1)
    elif uncertain_policy == "as_normal":
        llm_pred = llm_pred.where(~llm_pred.isna(), 0)
    elif uncertain_policy == "drop":
        pass
    else:
        raise ValueError(f"invalid uncertain_policy: {uncertain_policy}")

    note = f"uncertain_policy={uncertain_policy}, unresolved_rows={int(llm_pred.isna().sum())}"
    return llm_pred, note


def _merge_for_full_test(sm_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    if "TestIndex" in sm_df.columns and "TestIndex" in llm_df.columns:
        merged = sm_df.merge(
            llm_df[["TestIndex", "log_text", "llm_status", "llm_pred", "llm_level", "llm_reason", "llm_suggestion"]],
            on="TestIndex",
            how="left",
        )
        return merged

    # fallback: no stable key, try by common columns (less robust)
    join_cols = [c for c in ["Templates", "Label", "Pred", "AnomalyScore", "Uncertainty"] if c in sm_df.columns and c in llm_df.columns]
    if not join_cols:
        raise ValueError("无法对齐 full_test：缺少 TestIndex，且无可用共同列。请重新跑 SM/Inference 生成 TestIndex。")

    merged = sm_df.merge(
        llm_df[join_cols + [c for c in ["log_text", "llm_status", "llm_pred", "llm_level", "llm_reason", "llm_suggestion"] if c in llm_df.columns]],
        on=join_cols,
        how="left",
    )
    return merged


def evaluate(
    dataset: str,
    scope: str = "full_test",
    uncertain_policy: str = "fallback_sm",
    sm_csv: Optional[str] = None,
    llm_csv: Optional[str] = None,
    save_json: bool = True,
):
    llm_path = _resolve_results_file(dataset, "llm_second_pass_high_uncertain.csv", llm_csv)
    llm_df = pd.read_csv(llm_path)

    if "Label" not in llm_df.columns or "Pred" not in llm_df.columns:
        raise ValueError(f"{llm_path} 缺少 Label/Pred 列")

    if scope == "high_uncertain":
        eval_df = llm_df.copy()
    elif scope == "full_test":
        sm_path = _resolve_results_file(dataset, "sm_test_predictions.csv", sm_csv)
        sm_df = pd.read_csv(sm_path)
        if "Label" not in sm_df.columns or "Pred" not in sm_df.columns:
            raise ValueError(f"{sm_path} 缺少 Label/Pred 列")
        eval_df = _merge_for_full_test(sm_df=sm_df, llm_df=llm_df)
    else:
        raise ValueError("scope must be high_uncertain/full_test")

    y_true = _to_binary(eval_df["Label"])
    sm_pred = _to_binary(eval_df["Pred"])
    sm_score = pd.to_numeric(eval_df["AnomalyScore"], errors="coerce") if "AnomalyScore" in eval_df.columns else None

    llm_pred, llm_note = _build_llm_pred(eval_df, uncertain_policy=uncertain_policy, fallback_sm_pred=sm_pred)
    final_pred = llm_pred.where(~llm_pred.isna(), sm_pred)

    sm_metrics = _compute_metrics(y_true, sm_pred, score=sm_score)
    llm_metrics = _compute_metrics(y_true, llm_pred, score=None)
    cascade_metrics = _compute_metrics(y_true, final_pred, score=sm_score)

    llm_rows = int(eval_df["llm_status"].notna().sum()) if "llm_status" in eval_df.columns else 0
    unique_calls = int(eval_df["log_text"].dropna().nunique()) if "log_text" in eval_df.columns else llm_rows
    total_rows = len(eval_df)
    call_ratio = llm_rows / total_rows if total_rows > 0 else 0.0
    save_ratio = 1.0 - (unique_calls / total_rows) if total_rows > 0 else 0.0

    report = {
        "dataset": dataset,
        "scope": scope,
        "rows": total_rows,
        "llm_rows_covered": llm_rows,
        "llm_coverage_ratio": call_ratio,
        "unique_llm_calls_estimated": unique_calls,
        "llm_call_saving_ratio": save_ratio,
        "notes": {
            "llm_pred_build": llm_note,
            "cascade_rule": "final_pred = llm_pred(if available) else sm_pred",
        },
        "metrics": {
            "sm": sm_metrics,
            "llm": llm_metrics,
            "cascade": cascade_metrics,
            "delta_cascade_vs_sm": {
                "precision": cascade_metrics["precision"] - sm_metrics["precision"],
                "recall": cascade_metrics["recall"] - sm_metrics["recall"],
                "f1": cascade_metrics["f1"] - sm_metrics["f1"],
                "accuracy": cascade_metrics["accuracy"] - sm_metrics["accuracy"],
            },
        },
    }

    print("\n========== Cascade Evaluation ==========")
    print(f"dataset                : {dataset}")
    print(f"scope                  : {scope}")
    print(f"rows                   : {total_rows}")
    print(f"llm covered rows       : {llm_rows} ({call_ratio:.2%})")
    print(f"estimated unique calls : {unique_calls}")
    print(f"call saving ratio      : {save_ratio:.2%}")
    print(f"note                   : {llm_note}")

    def _line(name: str, m: Dict[str, float]) -> str:
        return (
            f"{name:<8} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} Acc={m['accuracy']:.4f} "
            f"Cov={m['coverage']:.2%} N={m['support']}"
        )

    print(_line("SM", sm_metrics))
    print(_line("LLM", llm_metrics))
    print(_line("CASCADE", cascade_metrics))
    print(
        "DELTA    "
        f"dP={report['metrics']['delta_cascade_vs_sm']['precision']:+.4f} "
        f"dR={report['metrics']['delta_cascade_vs_sm']['recall']:+.4f} "
        f"dF1={report['metrics']['delta_cascade_vs_sm']['f1']:+.4f} "
        f"dAcc={report['metrics']['delta_cascade_vs_sm']['accuracy']:+.4f}"
    )

    if save_json:
        out = Path("outputs") / dataset / "results" / f"cascade_eval_report_{scope}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"saved report           : {out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SM/LLM/cascade on high-uncertain or full-test scope.")
    parser.add_argument("--dataset", required=True, help="Spirit/BGL/HDFS/Thunderbird")
    parser.add_argument("--scope", default="full_test", choices=["full_test", "high_uncertain"])
    parser.add_argument(
        "--uncertain-policy",
        default="fallback_sm",
        choices=["fallback_sm", "as_anomaly", "as_normal", "drop"],
        help="How to handle LLM uncertain/unknown rows.",
    )
    parser.add_argument("--sm-csv", default=None, help="override SM full-test csv path")
    parser.add_argument("--llm-csv", default=None, help="override LLM second-pass csv path")
    parser.add_argument("--no-save", action="store_true", help="do not save json report")
    args = parser.parse_args()

    evaluate(
        dataset=args.dataset,
        scope=args.scope,
        uncertain_policy=args.uncertain_policy,
        sm_csv=args.sm_csv,
        llm_csv=args.llm_csv,
        save_json=not args.no_save,
    )
