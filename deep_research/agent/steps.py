"""
推理步骤 (Reasoning Steps)
==========================

设计意图
--------
把 Agent 的每一个"动脑子"的环节封装成独立、可单测的函数。这样做的好处:
- 每个步骤的提示词集中管理,调 prompt 时一目了然。
- 步骤之间通过 state 通信,职责单一,易于替换和评估。

包含五个步骤:
1. plan_subquestions   —— 把研究问题拆成子问题(SMART 档,低温)
2. refine_query        —— 根据已有发现改写下一次搜索查询(SMART 档)
3. reflect_subquestion —— 判断某子问题信息是否充分(SMART 档,终止条件之一)
4. reflect_global_gaps —— 全局反思,找出尚未覆盖的缺口
5. generate_report     —— 综合所有发现,生成带引用的报告(SMART 档)

防幻觉的核心机制都在 generate_report 里,见该函数注释。
"""

from __future__ import annotations

from ..llm.client import LLMClient, ModelTier
from ..tools.fetch import Finding
from .state import ResearchState


def plan_subquestions(llm: LLMClient, question: str, max_subq: int = 5) -> list[str]:
    """把一个开放式研究问题拆解成若干相对独立的子问题。

    设计要点:
    - 用 SMART 档,因为拆解质量直接决定整个研究的覆盖面,值得用好模型。
    - 温度设 0,要稳定可复现的拆解。
    - 要求返回 JSON 数组,便于程序解析(见 complete_json 的健壮解析)。
    - few-shot 风格的指令引导模型拆出"可被搜索回答"的具体子问题,
      而不是又一个大而空的问题。
    """
    prompt = (
        f"你是一个研究规划专家。请把下面这个研究问题拆解成 {max_subq} 个以内、"
        f"相对独立、每个都能通过网络搜索回答的具体子问题。\n\n"
        f"研究问题:{question}\n\n"
        f'以 JSON 数组返回,例如:["子问题1", "子问题2"]。'
        f"子问题要具体、可检索,避免空泛。"
    )
    data = llm.complete_json(prompt, tier=ModelTier.SMART, temperature=0.0)
    # 防御性处理:确保拿到的是字符串列表,且不超过上限。
    if isinstance(data, dict):
        # 有时模型会返回 {"sub_questions": [...]} 之类的包装
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise ValueError(f"规划器返回格式异常: {data!r}")
    return [str(x) for x in data][:max_subq]


def refine_query(llm: LLMClient, sub_question: str, existing: list[Finding]) -> str:
    """根据已有发现,为某子问题生成下一次搜索查询。

    第一次搜索时 existing 为空,直接基于子问题生成查询;
    后续搜索时,告诉模型"我们已经知道了什么",让它生成能补充新角度的查询,
    避免反复搜到相同结果(这也是减少无效循环的手段之一)。
    """
    if not existing:
        prompt = (
            f"请为下面的子问题生成一个简洁有效的网络搜索查询(英文优先,更易搜到资料),"
            f"只返回查询词本身,不要解释。\n\n子问题:{sub_question}"
        )
    else:
        known = "\n".join(f"- {f.content}" for f in existing)
        prompt = (
            f"针对子问题:{sub_question}\n"
            f"我们已经了解到:\n{known}\n\n"
            f"请生成一个新的搜索查询,以补充我们尚未覆盖的角度。"
            f"只返回查询词本身,不要解释。"
        )
    query = llm.complete(prompt, tier=ModelTier.SMART, temperature=0.3)
    # 去掉可能的引号包裹
    return query.strip().strip('"').strip()


def reflect_subquestion(llm: LLMClient, sub_question: str, findings: list[Finding]) -> bool:
    """反思:针对某子问题,现有发现是否已经足够回答它?

    返回 True 表示"够了,可以停止对该子问题的搜索"。
    这是 Agent 的核心"终止条件"之一,直接关系到成本与质量的平衡:
    - 判得太松 → 信息不足,报告质量差。
    - 判得太严 → 反复搜索,烧钱还可能死循环(所以还有硬性次数上限兜底)。

    实现上让模型返回 {"enough": true/false, "reason": "..."},
    reason 字段纯粹是为了可观测性(写进 trace,方便调试和面试演示)。
    """
    if not findings:
        return False
    known = "\n".join(f"- {f.content}" for f in findings)
    prompt = (
        f"子问题:{sub_question}\n"
        f"目前收集到的信息:\n{known}\n\n"
        f"这些信息是否已经足以充分回答该子问题?"
        f'以 JSON 返回:{{"enough": true 或 false, "reason": "简短理由"}}。'
    )
    data = llm.complete_json(prompt, tier=ModelTier.SMART, temperature=0.0)
    return bool(data.get("enough", False)) if isinstance(data, dict) else False


def reflect_global_gaps(llm: LLMClient, state: ResearchState) -> list[str]:
    """全局反思:纵观所有子问题的发现,是否还有重要缺口需要补搜?

    返回一个"补充子问题"列表(可能为空)。这一步让 Agent 具备"退一步看全局"
    的能力,而不是机械地走完预设子问题就收工。面试时可强调:这是
    "规划-执行-反思"循环里第二层反思,提升报告的完整性。
    """
    summary_lines = []
    for subq in state.sub_questions:
        n = len(state.findings.get(subq, []))
        summary_lines.append(f"- 子问题「{subq}」:已有 {n} 条发现")
    overview = "\n".join(summary_lines)
    prompt = (
        f"原始研究问题:{state.original_question}\n"
        f"当前各子问题的覆盖情况:\n{overview}\n\n"
        f"为了全面回答原始问题,是否还有尚未覆盖的重要角度?"
        f'以 JSON 数组返回需要补充的子问题(最多 2 个),没有则返回 []。'
    )
    data = llm.complete_json(prompt, tier=ModelTier.SMART, temperature=0.0)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        return []
    return [str(x) for x in data][:2]


def generate_report(llm: LLMClient, state: ResearchState) -> str:
    """综合所有发现,生成带引用的最终报告。

    ★ 防幻觉 / 可溯源的核心机制(面试必讲)★
    ------------------------------------------
    我们不是让模型"凭记忆写报告",而是:
    1. 给每条 finding 编一个引用序号 [1] [2] ...,并附上其真实 source_url。
    2. 在提示词里**强约束**:报告中的每个关键论断都必须基于所提供的发现,
       并在句末用 [序号] 标注来源;不得编造发现里没有的事实。
    3. 报告末尾自动附上"参考来源"列表,序号对应真实 URL。

    这样,报告里的每个引用都指向一个我们**真实抓取过**的网页,而不是模型
    幻想出来的链接。这是把"可溯源"从口号变成机制的关键。
    """
    findings = state.all_findings()
    if not findings:
        return "未能收集到足够的信息来生成报告。请检查搜索工具是否正常,或更换研究问题。"

    # 给每条发现编号,构造"带编号的材料"
    numbered = []
    sources = []
    for i, f in enumerate(findings, start=1):
        numbered.append(f"[{i}] {f.content}(来源:{f.source_title})")
        sources.append(f"[{i}] {f.source_title} - {f.source_url}")
    material = "\n".join(numbered)
    source_list = "\n".join(sources)

    prompt = (
        f"你是一位严谨的研究分析师。请基于下面带编号的研究发现,撰写一份结构清晰的"
        f"中文研究报告,回答这个问题:{state.original_question}\n\n"
        f"研究发现(每条带编号):\n{material}\n\n"
        f"写作要求:\n"
        f"1. 报告要有清晰的小标题分点论述。\n"
        f"2. 每个关键论断后面用 [编号] 标注它依据的发现,可引用多个如 [1][3]。\n"
        f"3. 严禁编造发现中不存在的事实或数据。\n"
        f"4. 如果发现之间存在矛盾,要明确指出。\n"
        f"5. 不要在正文里写参考来源列表,我会另外附上。"
    )
    body = llm.complete(prompt, tier=ModelTier.SMART, temperature=0.4, max_tokens=2500)

    # 自动拼接真实来源列表
    return f"{body}\n\n---\n\n## 参考来源\n\n{source_list}"
