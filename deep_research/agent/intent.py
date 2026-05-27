"""
对话意图分类 (Intent Classification)
====================================

★ 这是多轮研究 Agent 的灵魂,务必能讲清楚 ★

为什么需要意图分类
------------------
单轮研究很简单:输入问题 → 跑流程 → 出报告。但一旦支持多轮对话,核心难点
立刻浮现:**用户这一轮说的话,到底想干嘛?**

同样一句"那性能方面呢?",在不同上下文里含义完全不同:
- 如果上一轮刚出了一份向量数据库对比报告 → 这是对报告的**追问(FOLLOW_UP)**,
  也许已有发现就能回答,根本不用重新搜索。
- 如果用户想要更详尽的性能数据 → 这是要求**深入(DEEPEN)**,该触发针对性检索。

如果不做意图分类,把每句话都当成"全新研究"重新跑一遍完整流程,既慢又贵,
而且完全没利用上多轮对话积累的上下文 —— 那就白瞎了"多轮"。

四种意图
--------
- NEW_RESEARCH:全新的研究主题,跑完整的规划-执行-反思流程。
- DEEPEN:针对已有研究的某个方面深入挖掘,触发定向的补充检索。
- FOLLOW_UP:基于已积累的发现就能回答的追问,直接生成回答,不必重新搜索(省钱省时)。
- REFINE:换个角度或补充维度,在已有研究基础上追加新的子问题。

面试谈资
--------
"多轮研究 Agent 最难的不是'能聊',而是'听懂用户这轮要什么'。我用一个轻量 LLM
调用做意图路由,把追问直接用已有发现回答、把深入才触发新检索 —— 这样既省成本,
又真正利用了对话上下文。这是有状态 Agent 区别于无状态问答的关键。"
"""

from __future__ import annotations

from enum import Enum

from ..llm.client import LLMClient, ModelTier


class Intent(str, Enum):
    NEW_RESEARCH = "NEW_RESEARCH"  # 全新研究主题
    DEEPEN = "DEEPEN"              # 深入某个已研究的方面
    FOLLOW_UP = "FOLLOW_UP"        # 基于已有发现就能回答
    REFINE = "REFINE"             # 换角度/补充新维度


def classify_intent(
    llm: LLMClient,
    user_message: str,
    *,
    has_prior_research: bool,
    prior_topic: str | None,
    sub_questions: list[str] | None = None,
) -> Intent:
    """判断用户这一轮消息的意图。

    设计要点:
    - 如果还没有任何历史研究(has_prior_research=False),那必然是 NEW_RESEARCH,
      直接短路返回,省一次 LLM 调用。这是"能用规则就别用模型"的成本意识。
    - 有历史时才用 LLM 分类,并把上一轮主题和子问题作为上下文喂进去,
      让分类更准。
    - 用 FAST 档:意图分类是简单的短文本分类任务,小模型足矣(又一处模型分级)。
    """
    # 规则短路:没有历史,只能是新研究。
    if not has_prior_research:
        return Intent.NEW_RESEARCH

    sq_text = ""
    if sub_questions:
        sq_text = "已研究的子问题:\n" + "\n".join(f"- {s}" for s in sub_questions)

    prompt = (
        f"你是一个对话意图分类器。当前正在进行一个多轮研究对话。\n"
        f"上一轮研究的主题是:{prior_topic}\n"
        f"{sq_text}\n\n"
        f"用户最新的消息是:「{user_message}」\n\n"
        f"请判断用户这条消息的意图,从以下四类中选一个:\n"
        f"- NEW_RESEARCH:这是一个与之前完全不同的全新研究主题。\n"
        f"- DEEPEN:用户想就之前研究的某个具体方面挖得更深、要更多细节。\n"
        f"- FOLLOW_UP:用户在追问,基于已有研究内容应该就能回答,不需要重新检索。\n"
        f"- REFINE:用户想换一个角度,或补充一个之前没覆盖的新维度。\n\n"
        f'只返回 JSON:{{"intent": "上述四个之一", "reason": "简短理由"}}。'
    )
    data = llm.complete_json(prompt, tier=ModelTier.FAST, temperature=0.0)
    raw = data.get("intent", "") if isinstance(data, dict) else ""
    try:
        return Intent(raw)
    except ValueError:
        # 模型返回了意料外的值,保守降级为 FOLLOW_UP(最便宜的处理路径)。
        return Intent.FOLLOW_UP


def answer_follow_up(llm: LLMClient, user_message: str, context_findings: str) -> str:
    """FOLLOW_UP 意图的处理:直接基于已有发现回答,不重新检索。

    这是多轮对话省钱的关键路径 —— 很多追问("第二点能再解释下吗")根本不需要
    上网搜,已经收集的发现里就有答案。同样强约束"只基于已有材料,不编造"。
    """
    prompt = (
        f"基于下面已经收集到的研究发现,回答用户的追问。"
        f"只使用这些材料,不要编造材料里没有的信息;"
        f"如果材料不足以回答,就如实说明并建议用户让你深入检索。\n\n"
        f"已有研究发现:\n{context_findings}\n\n"
        f"用户追问:{user_message}"
    )
    return llm.complete(prompt, tier=ModelTier.SMART, temperature=0.3)
