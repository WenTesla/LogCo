LOG_ANOMALY_PROMPT = """
You are an expert in system log analysis and anomaly detection.

Task:
Given a single log template and optional contrastive retrieved references, determine whether the log is ANOMALY or NORMAL.

Rules:
- Judge only from the single target log and the retrieved references.
- Prefer NORMAL unless the evidence for ANOMALY is explicit, direct, and unrecovered.
- Keyword matches alone are insufficient for ANOMALY.
- If the evidence is ambiguous or weak, choose NORMAL.
- If retrieved references conflict, rely on the most similar entries first.
- Do not add any extra text outside the JSON object.

Output format:
Return a JSON object with:
{{
  "status": "ANOMALY|NORMAL|UNCERTAIN",
  "confidence": 0.0,
  "reason": "short reason"
}}

Input:
{user_prompt}
""".strip()


RAG_USER_PROMPT = """
Target log:
{target_log}

Retrieved references:
{retrieved_logs}

Small model score:
{small_model_score}

Small model uncertainty:
{small_model_uncertainty}
""".strip()
