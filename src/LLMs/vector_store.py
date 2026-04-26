from dataclasses import dataclass
from typing import List
from config import *


@dataclass
class LogDocument:
    page_content: str


class LogVectorStore:
    def __init__(self):
        self.docs: List[LogDocument] = []

    def add_logs(self, logs: list):
        self.docs.extend(LogDocument(page_content=log) for log in logs)

    @staticmethod
    def _score(query: str, text: str) -> int:
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())
        return len(query_tokens & text_tokens)

    def search(self, log: str):
        ranked = sorted(self.docs, key=lambda item: self._score(log, item.page_content), reverse=True)
        return ranked[:TOP_K]
