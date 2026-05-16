import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from config import RULES_DIR, RULE_TOP_K


@dataclass(frozen=True)
class DecisionRule:
    rule_id: str
    dataset: str
    priority: int
    decision: str
    keywords: List[str]
    condition: str
    rationale: str

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.rule_id,
                self.dataset,
                self.decision,
                " ".join(self.keywords),
                self.condition,
                self.rationale,
            ]
        )


class RuleStore:
    def __init__(self, dataset: str, rules_dir: str | Path = RULES_DIR):
        self.dataset = dataset
        self.rules_dir = Path(rules_dir)
        self.rules = self._load_rules()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_<>\-]+", str(text).lower()))

    @staticmethod
    def _keyword_hits(log_text: str, keywords: Iterable[str]) -> List[str]:
        lowered = str(log_text).lower()
        hits = []
        for keyword in keywords:
            keyword_text = str(keyword).strip()
            if not keyword_text:
                continue
            keyword_lower = keyword_text.lower()
            if len(keyword_lower) <= 3 and re.fullmatch(r"[a-zA-Z0-9_]+", keyword_lower):
                matched = re.search(rf"\b{re.escape(keyword_lower)}\b", lowered) is not None
            else:
                matched = keyword_lower in lowered
            if matched:
                hits.append(keyword_text)
        return hits

    def _load_rules(self) -> List[DecisionRule]:
        dataset_path = self.rules_dir / f"{self.dataset}.jsonl"
        common_path = self.rules_dir / "common.jsonl"
        paths = [common_path, dataset_path]

        rules = []
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    text = line.strip()
                    if not text or text.startswith("#"):
                        continue
                    try:
                        item = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"规则文件格式错误: {path}:{line_no}: {exc}") from exc
                    rules.append(
                        DecisionRule(
                            rule_id=str(item["rule_id"]),
                            dataset=str(item.get("dataset", self.dataset)),
                            priority=int(item.get("priority", 0)),
                            decision=str(item["decision"]).upper(),
                            keywords=[str(k) for k in item.get("keywords", [])],
                            condition=str(item.get("condition", "")),
                            rationale=str(item.get("rationale", "")),
                        )
                    )
        return rules

    def search(self, log_text: str, top_k: int = RULE_TOP_K) -> List[dict]:
        if not self.rules:
            return []

        query_tokens = self._tokens(log_text)
        scored = []
        for rule in self.rules:
            hits = self._keyword_hits(log_text, rule.keywords)
            if not hits:
                continue
            rule_tokens = self._tokens(rule.text)
            token_overlap = len(query_tokens & rule_tokens)
            score = rule.priority * 10 + len(hits) * 10 + token_overlap
            scored.append((score, hits, rule))

        scored.sort(key=lambda item: (item[0], item[2].priority), reverse=True)
        return [
            {
                "ref": f"D{i}",
                "rule_id": rule.rule_id,
                "decision": rule.decision,
                "priority": rule.priority,
                "matched_keywords": hits,
                "keywords": rule.keywords,
                "condition": rule.condition,
                "rationale": rule.rationale,
                "score": round(score, 4),
            }
            for i, (score, hits, rule) in enumerate(scored[:top_k], start=1)
        ]


def format_rules_for_prompt(rules: List[dict]) -> str:
    if not rules:
        return "No dataset-specific decision rules were retrieved."

    lines = []
    for rule in rules:
        matched = ", ".join(rule["matched_keywords"]) if rule["matched_keywords"] else "N/A"
        keywords = ", ".join(rule["keywords"]) if rule["keywords"] else "N/A"
        lines.append(
            f"[{rule['ref']}] decision={rule['decision']}, priority={rule['priority']}, "
            f"matched_keywords={matched}, keywords={keywords}\n"
            f"condition: {rule['condition']}\n"
            f"rationale: {rule['rationale']}"
        )
    return "\n\n".join(lines)
