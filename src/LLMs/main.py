import json

from llm_client import get_llm
from prompt_templates import LOG_ANOMALY_PROMPT
from vector_store import LogVectorStore
from config import NORMAL_LOGS

# 初始化
llm = get_llm()
vector_db = LogVectorStore()


# 构建知识库
vector_db.add_logs(NORMAL_LOGS)
print("✅ 正常日志知识库已构建")


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

    return {
        "status": parsed.get("status", "unknown"),
        "level": parsed.get("level", "unknown"),
        "reason": parsed.get("reason", text),
        "suggestion": parsed.get("suggestion", "N/A"),
    }


def detect(log: str):
    print("\n" + "="*60)
    print(f"日志：{log}")

    # 1. 检索相似日志
    docs = vector_db.search(log)
    context = "\n".join([d.page_content for d in docs])

    # 2. 构造提示词
    prompt = LOG_ANOMALY_PROMPT.format(
        retrieved_logs=context,
        target_log=log
    )

    # 3. LLM 调用
    result = llm.invoke(prompt)

    # 4. 解析
    parsed = _parse_result(_extract_content(result))
    print(f"异常状态：{parsed['status']}")
    print(f"等级：{parsed['level']}")
    print(f"原因：{parsed['reason']}")
    print(f"建议：{parsed['suggestion']}")
    return parsed

# 测试
if __name__ == "__main__":
    detect("INFO Server started successfully")
    detect("ERROR Connection timeout to database")
    detect("EXCEPTION NullPointerException at line 99")
