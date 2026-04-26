import json
import requests
from config import *


class MockLLM:
    def invoke(self, prompt: str):
        text = prompt.upper()
        is_anomaly = any(token in text for token in ["ERROR", "EXCEPTION", "FAIL", "TIMEOUT"])
        payload = {
            "status": "anomaly" if is_anomaly else "normal",
            "level": "high" if is_anomaly else "low",
            "reason": "检测到异常关键词" if is_anomaly else "与知识库正常模式相似",
            "suggestion": "排查异常堆栈与依赖服务" if is_anomaly else "持续监控",
        }
        return type("MockResponse", (), {"content": json.dumps(payload, ensure_ascii=False)})()


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
        return type("HTTPResponse", (), {"content": content})()


def get_llm():
    """获取 LLM 实例"""
    if LLM_TYPE == "mock":
        return MockLLM()
    if LLM_TYPE == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("缺少 OPENAI_API_KEY，请先设置环境变量。")
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=OPENAI_MODEL,
                temperature=0.0
            )
        except ModuleNotFoundError:
            return HTTPChatLLM(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=OPENAI_MODEL,
                temperature=0.0,
            )
    elif LLM_TYPE == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0
        )
    raise ValueError("LLM_TYPE 必须是 mock / openai / ollama")


if __name__ == "__main__":
    llm = get_llm()
    print(f"✅ LLM 模型加载成功：{LLM_TYPE}")
