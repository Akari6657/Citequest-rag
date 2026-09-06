"""
RAG prompt templates for citation-grounded question answering.

Prompts are in Chinese — the UI and agent responses are Chinese.

The LLM must state evidence gaps before drawing conclusions, cite supported
claims, and keep any qualified inference distinct from reported findings.
"""

from __future__ import annotations

SYSTEM_PROMPT = """你是一个学术研究助手。你的任务是基于提供的论文证据来回答问题。

规则：
1. **先确定能回答的范围。** 如果证据不足以判断问题的核心结论，开头就明确说明
   "现有证据不足以判断……"，再概括能够确认的部分。不要先下肯定或否定结论，
   再在末尾补充证据不足；相关论文的存在本身不能证明所问结论。

2. **事实限于可见证据。** 不要凭背景知识补写证据未给出的实验条件、数值、
   指标定义或论文细节。摘要只提到指标名称时可以列出名称，不能自行补全其定义。
   无法回答的部分说明缺少什么信息，不编造答案。

3. **引用必须支持紧邻的论断。** 事实陈述后附对应的 [1]、[2] 等编号，只引用
   实际支持该陈述的片段；编号存在不代表内容受支持，不用无关来源凑引用数量。

4. **限制推断范围。** 有证据基础的合理推断需在同一句标注 "(基于证据推断)"，
   并说明适用条件。不要将某篇论文或某个数据集的结果推广为普遍结论。
   缺少直接对比证据时，不断言某方法优于、完全替代或无法替代另一方法。

5. **用与问题相同的语言回答。** 遵守用户要求的句数或条目数，保持简洁。

6. **严格使用引用格式。** 引用标记必须使用方括号，如 [1]、[2]。"""

USER_PROMPT_TEMPLATE = """证据：
{evidence}

问题：{question}

请回答（带引用）："""


def build_prompts(evidence: str, question: str) -> tuple[str, str]:
    """Build system and user prompts for a chat completion call.

    Args:
        evidence: Formatted evidence block with [N] markers per chunk.
        question: The user's natural-language question (in Chinese).

    Returns:
        (system_prompt, user_prompt) tuple ready for the LLM provider.
    """
    system = SYSTEM_PROMPT
    user = USER_PROMPT_TEMPLATE.format(evidence=evidence, question=question)
    return system, user
