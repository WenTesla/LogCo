import os

# ===================== LLM 配置 =====================
LLM_TYPE = os.getenv("LLM_TYPE", "openai")  # mock / openai / ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K = 5

# ===================== 正常日志样本 =====================
NORMAL_LOGS = [
    "INFO Server started successfully",
    "INFO Request processed in 12ms",
    "INFO Database connection established",
    "INFO User login success",
    "INFO Health check passed"
]
