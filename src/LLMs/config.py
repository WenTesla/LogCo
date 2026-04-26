import os

# ===================== LLM 配置 =====================
LLM_TYPE = os.getenv("LLM_TYPE", "openai")  # mock / openai / ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-f273396839ba400eb48dff31f034ee0d")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K = 5

# ===================== RAG 配置 =====================
DATASET = os.getenv("DATASET", "BGL")
# BGE3_MODEL_NAME = os.getenv("BGE3_MODEL_NAME", "BAAI/bge-m3")
BGE3_MODEL_NAME = os.getenv("BGE3_MODEL_NAME", "BAAI/bge-small-en-v1.5")
VECTOR_STORE_DIRNAME = os.getenv("VECTOR_STORE_DIRNAME", "faiss_bge3")
RAG_SPLIT_MODE = os.getenv("RAG_SPLIT_MODE", "ordered")  # ordered / random
RAG_TRAIN_RATIO = float(os.getenv("RAG_TRAIN_RATIO", "0.7"))
RAG_RANDOM_SEED = int(os.getenv("RAG_RANDOM_SEED", "42"))
RAG_USE_TRAIN_ONLY = os.getenv("RAG_USE_TRAIN_ONLY", "true").lower() == "true"
