import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _templates_to_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value if str(x).strip())
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return " ".join(str(x) for x in parsed if str(x).strip())
    except Exception:
        pass
    return text


def _split_indices(n: int, train_ratio: float, mode: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    train_size = int(n * train_ratio)
    if train_size <= 0 or train_size >= n:
        raise ValueError(f"非法 train_ratio={train_ratio}，当前样本数={n}")

    if mode == "ordered":
        train_idx = np.arange(0, train_size)
        test_idx = np.arange(train_size, n)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        train_idx = np.sort(perm[:train_size])
        test_idx = np.sort(perm[train_size:])
    else:
        raise ValueError("mode 必须是 ordered 或 random")
    return train_idx, test_idx


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _label_to_binary(series: pd.Series) -> pd.Series:
    """将标签列稳健映射为二值：0=正常, 1=异常。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0) > 0).astype(int)

    s = series.astype(str).str.strip()
    lower = s.str.lower()
    normal_tokens = {"", "-", "0", "normal", "false", "benign", "ok"}
    return (~lower.isin(normal_tokens)).astype(int)


def _resolve_default_csv_path(dataset: str) -> Path:
    candidates = [
        Path("outputs") / dataset / "grouped_logs.csv",
        Path("output") / dataset / "grouped_logs.csv",
        Path("inputs") / dataset / "grouped_logs.csv",
        Path("input") / dataset / "grouped_logs.csv",
        Path("inputs") / dataset / "structured.csv",
        Path("input") / dataset / "structured.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "未找到默认 CSV。已尝试: "
        + ", ".join(str(p) for p in candidates)
    )


def _build_token_counter(texts: List[str]) -> Counter:
    c = Counter()
    for t in texts:
        tokens = t.split()
        c.update(tokens)
    return c


def _js_divergence_from_counters(c1: Counter, c2: Counter) -> float:
    vocab = sorted(set(c1.keys()) | set(c2.keys()))
    if not vocab:
        return 0.0
    p = np.array([c1.get(k, 0) for k in vocab], dtype=np.float64)
    q = np.array([c2.get(k, 0) for k in vocab], dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log2((p + 1e-12) / (m + 1e-12)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log2((q + 1e-12) / (m + 1e-12)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def evaluate_split(df: pd.DataFrame, train_ratio: float, mode: str, seed: int) -> dict:
    n = len(df)
    train_idx, test_idx = _split_indices(n, train_ratio, mode, seed)
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_texts = train_df["log_text"].tolist()
    test_texts = test_df["log_text"].tolist()

    train_templates = set(train_texts)
    test_templates = set(test_texts)

    overlap_templates = train_templates & test_templates
    unseen_templates = test_templates - train_templates

    train_token_counter = _build_token_counter(train_texts)
    test_token_counter = _build_token_counter(test_texts)
    train_vocab = set(train_token_counter.keys())
    test_vocab = set(test_token_counter.keys())

    unseen_test_tokens = test_vocab - train_vocab
    unseen_test_token_count = sum(test_token_counter[t] for t in unseen_test_tokens)
    total_test_tokens = sum(test_token_counter.values())

    label_col_exists = "Label" in df.columns
    if label_col_exists:
        train_pos = int(_label_to_binary(train_df["Label"]).sum())
        test_pos = int(_label_to_binary(test_df["Label"]).sum())
        train_pos_rate = _safe_div(train_pos, len(train_df))
        test_pos_rate = _safe_div(test_pos, len(test_df))
    else:
        train_pos_rate = None
        test_pos_rate = None

    return {
        "mode": mode,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "template_overlap_rate": _safe_div(len(overlap_templates), len(test_templates)),
        "test_unseen_template_rate": _safe_div(len(unseen_templates), len(test_templates)),
        "token_vocab_overlap_rate": _safe_div(len(train_vocab & test_vocab), len(test_vocab)),
        "test_unseen_token_ratio": _safe_div(unseen_test_token_count, total_test_tokens),
        "token_js_divergence": _js_divergence_from_counters(train_token_counter, test_token_counter),
        "train_pos_rate": train_pos_rate,
        "test_pos_rate": test_pos_rate,
    }


def _fmt_pct(x):
    if x is None:
        return "N/A"
    return f"{x * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="验证 ordered 划分是否引入 OOD")
    parser.add_argument("--dataset", default="BGL", help="数据集名称，例如 BGL/HDFS/Spirit/Thunderbird")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="CSV 路径；不传时自动检测 output(s)/input(s) 下的 grouped_logs.csv 或 structured.csv",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--random-seed", type=int, default=42, help="随机划分种子")
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        csv_path = _resolve_default_csv_path(args.dataset)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Templates" not in df.columns and "EventTemplate" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 Templates 或 EventTemplate 列")
    df = df.copy()
    if "Templates" in df.columns:
        df["log_text"] = df["Templates"].apply(_templates_to_text)
    else:
        df["log_text"] = df["EventTemplate"].apply(_templates_to_text)

    ordered_result = evaluate_split(df, args.train_ratio, "ordered", args.random_seed)
    random_result = evaluate_split(df, args.train_ratio, "random", args.random_seed)

    print("\n================ OOD Split Validation ================")
    print(f"CSV            : {csv_path}")
    print(f"Samples        : {len(df)}")
    print(f"Train Ratio    : {args.train_ratio}")
    print(f"Random Seed    : {args.random_seed}")

    for r in [ordered_result, random_result]:
        print(f"\n[{r['mode'].upper()}]")
        print(f"Train/Test                     : {r['train_size']} / {r['test_size']}")
        print(f"Test模板在Train出现比例         : {_fmt_pct(r['template_overlap_rate'])}")
        print(f"Test未见模板比例                : {_fmt_pct(r['test_unseen_template_rate'])}")
        print(f"Test词表在Train覆盖率           : {_fmt_pct(r['token_vocab_overlap_rate'])}")
        print(f"Test未见词Token占比             : {_fmt_pct(r['test_unseen_token_ratio'])}")
        print(f"Train/Test词分布JS散度          : {r['token_js_divergence']:.4f}")
        if r["train_pos_rate"] is not None:
            print(f"Train异常率 / Test异常率        : {_fmt_pct(r['train_pos_rate'])} / {_fmt_pct(r['test_pos_rate'])}")

    print("\n---------------- Interpretation ----------------")
    print("若 ORDERED 相比 RANDOM 出现：")
    print("1) 更高的 Test未见模板比例 / 未见词占比")
    print("2) 更低的 模板重叠率 / 词表覆盖率")
    print("3) 更高的 JS 散度")
    print("则可认为 ORDERED 划分更可能存在分布外(OOD)问题。")


if __name__ == "__main__":
    main()
