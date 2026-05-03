import argparse
import ast
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import requests

from config import (
    DATASET,
    LLM_TYPE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)
from prompt_templates import Summarize_System_Instruction


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
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return type("HTTPResponse", (), {"content": content})()


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

        return OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )
    raise ValueError("LLM_TYPE 必须是 openai / ollama")


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


def _normalize_text_key(text: str) -> str:
    return " ".join(str(text).split())


def _normalize_label(value) -> int:
    text = str(value).strip().lower()
    if text in {"", "-", "0", "normal", "false", "benign", "ok"}:
        return 0
    try:
        return 1 if float(text) > 0 else 0
    except Exception:
        return 1


def _resolve_csv_path(dataset: str, csv_path: str | None) -> Path:
    if csv_path:
        return Path(csv_path)
    return Path("outputs") / dataset / "grouped_logs.csv"


def _sample_texts(texts: List[str], max_samples: int, seed: int) -> List[str]:
    if max_samples <= 0 or len(texts) <= max_samples:
        return texts
    return pd.Series(texts).sample(n=max_samples, random_state=seed).tolist()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_examples(title: str, texts: Iterable[str], max_chars: int) -> str:
    lines = [title]
    for idx, text in enumerate(texts, 1):
        lines.append(f"{idx}. {_truncate(text, max_chars)}")
    return "\n".join(lines)


def build_prompt(
    normal_texts: List[str],
    anomalous_texts: List[str],
) -> str:
    normal_block = _format_examples("NORMAL log sequences:", normal_texts, max_chars=400)
    anomaly_block = _format_examples("ANOMALOUS log sequences:", anomalous_texts, max_chars=400)
    return Summarize_System_Instruction.format(
        normal_log_sequences=normal_block,
        anomalous_log_sequences=anomaly_block,
    )


def summarize_system_instruction(
    dataset: str = DATASET,
    csv_path: str | None = None,
    output_json: str | None = None,
    dedup: bool = True,
    max_normal: int = 40,
    max_anomaly: int = 40,
    random_seed: int = 42,
):
    source_csv = _resolve_csv_path(dataset, csv_path)
    if not source_csv.exists():
        raise FileNotFoundError(f"未找到文件: {source_csv}")

    df = pd.read_csv(source_csv)
    if "Label" not in df.columns:
        raise ValueError(f"{source_csv} 缺少 Label 列")
    if "Templates" not in df.columns and "EventTemplate" not in df.columns:
        raise ValueError(f"{source_csv} 缺少 Templates 或 EventTemplate 列")

    text_col = "Templates" if "Templates" in df.columns else "EventTemplate"
    df = df.copy()
    df["log_text"] = df[text_col].apply(_templates_to_text)
    df["log_key"] = df["log_text"].apply(_normalize_text_key)
    df["is_anomaly"] = df["Label"].apply(_normalize_label)

    normal_df = df.loc[df["is_anomaly"] == 0, ["log_key", "log_text"]].copy()
    anomaly_df = df.loc[df["is_anomaly"] == 1, ["log_key", "log_text"]].copy()

    if dedup:
        normal_df = normal_df.drop_duplicates(subset=["log_key"], keep="first")
        anomaly_df = anomaly_df.drop_duplicates(subset=["log_key"], keep="first")

    normal_texts = normal_df["log_text"].tolist()
    anomaly_texts = anomaly_df["log_text"].tolist()
    if not normal_texts:
        raise ValueError("没有找到正常样本")
    if not anomaly_texts:
        raise ValueError("没有找到异常样本")

    normal_texts = _sample_texts(normal_texts, max_normal, random_seed)
    anomaly_texts = _sample_texts(anomaly_texts, max_anomaly, random_seed)

    prompt = build_prompt(normal_texts, anomaly_texts)
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    try:
        response_json = json.loads(response_text)
    except Exception:
        response_json = None

    result = {
        "prompt_name": "Summarize_System_Instruction",
        "dataset": dataset,
        "source_csv": str(source_csv),
        "dedup_enabled": dedup,
        "normal_count": int((df["is_anomaly"] == 0).sum()),
        "anomaly_count": int((df["is_anomaly"] == 1).sum()),
        "normal_unique_count": int(len(normal_df)),
        "anomaly_unique_count": int(len(anomaly_df)),
        "normal_duplicates_removed": int((df["is_anomaly"] == 0).sum() - len(normal_df)),
        "anomaly_duplicates_removed": int((df["is_anomaly"] == 1).sum() - len(anomaly_df)),
        "normal_examples_used": len(normal_texts),
        "anomaly_examples_used": len(anomaly_texts),
        "prompt": prompt,
        "response_text": response_text,
        "response_json": response_json,
    }

    out_path = Path(output_json) if output_json else Path("outputs") / dataset / "results" / "system_instruction_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已保存: {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="基于正常/异常样本生成语义化总结并保存为 JSON")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-normal", type=int, default=40)
    parser.add_argument("--max-anomaly", type=int, default=40)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    summarize_system_instruction(
        dataset=args.dataset,
        csv_path=args.csv_path,
        output_json=args.output_json,
        dedup=args.dedup,
        max_normal=args.max_normal,
        max_anomaly=args.max_anomaly,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
