"""
评估框架 (Evaluation Framework)
===============================

★ 这是让简历项目"经得起追问"的核心模块,务必能讲清楚 ★

为什么 eval 是初级和中级候选人的分水岭
--------------------------------------
面试官问"你怎么知道你的 Agent 好用?"时,"我跑了几个例子感觉不错"是最弱的回答。
强回答是:"我建了 N 个用例的评估集,定义了 X 个指标,某次改动让指标从 A% 提升到 B%。"
这个模块就是为了让你能说出后者。

本框架评估三个维度
------------------
1. **引用准确率 (citation accuracy)**:报告里标注 [n] 的论断,其引用的来源是否
   真实存在于发现列表中(而不是模型编的编号)。用程序直接校验,无需人工。
2. **覆盖度 (coverage)**:对每个测试问题,我们预先列出"理想报告应覆盖的关键点",
   用 LLM-as-judge 检查报告命中了几个。
3. **成本 (cost)**:平均每次研究消耗的 token 数与模型调用次数。

LLM-as-judge 的局限(主动暴露这个局限是成熟度的体现)
--------------------------------------------------
用模型当裁判有偏差(可能偏好啰嗦的答案、可能和被测模型同源导致"自我偏好")。
所以面试时要诚实地说:"我用 LLM-as-judge 做快速迭代,但知道它有偏差,
理想情况会引入人工标注做校准。"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..agent.state import ResearchState
from ..llm.client import LLMClient, ModelTier


@dataclass
class EvalCase:
    """一个评估用例。"""

    question: str
    # 理想报告应覆盖的关键点(人工预先定义),用于计算覆盖度。
    expected_points: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """单个用例的评估结果。"""

    question: str
    citation_accuracy: float  # 0~1
    coverage: float  # 0~1
    total_tokens: int
    llm_calls: int
    num_findings: int


def evaluate_citation_accuracy(state: ResearchState) -> float:
    """校验报告中的 [n] 引用编号是否都在合法范围内。

    报告生成时我们给了 len(findings) 条带编号的发现,所以合法编号是 1..N。
    如果报告里出现 [N+1] 这种越界编号,说明模型在编造引用 —— 直接判为不准确。
    这是一个**纯程序、零成本**的硬校验,非常适合自动化 eval。
    """
    findings = state.all_findings()
    n = len(findings)
    if n == 0 or not state.report:
        return 0.0

    cited = re.findall(r"\[(\d+)\]", state.report)
    # 只看正文部分的引用(参考来源列表里的编号是我们自己拼的,必然合法,排除掉)
    body = state.report.split("## 参考来源")[0]
    cited = re.findall(r"\[(\d+)\]", body)
    if not cited:
        return 0.0  # 报告没有任何引用,可溯源性为 0

    valid = sum(1 for c in cited if 1 <= int(c) <= n)
    return valid / len(cited)


def evaluate_coverage(llm: LLMClient, case: EvalCase, state: ResearchState) -> float:
    """用 LLM-as-judge 评估报告覆盖了多少预定义关键点。"""
    if not case.expected_points or not state.report:
        return 0.0
    points = "\n".join(f"{i+1}. {p}" for i, p in enumerate(case.expected_points))
    prompt = (
        f"下面是一份研究报告,以及它本应覆盖的关键点清单。请判断报告实际覆盖了"
        f"哪些关键点。\n\n报告:\n{state.report}\n\n"
        f"应覆盖的关键点:\n{points}\n\n"
        f'以 JSON 返回:{{"covered": [被覆盖关键点的编号列表]}}。'
    )
    data = llm.complete_json(prompt, tier=ModelTier.SMART, temperature=0.0)
    covered = data.get("covered", []) if isinstance(data, dict) else []
    return len(covered) / len(case.expected_points)


def run_eval(agent_factory, cases: list[EvalCase], judge_llm: LLMClient) -> list[CaseResult]:
    """对一组用例跑评估。

    `agent_factory` 是一个返回新 DeepResearchAgent 的可调用对象 —— 每个用例都用
    全新的 agent(以及全新的 token 计数器),保证成本统计互不干扰。

    返回每个用例的 CaseResult 列表;打印汇总指标。
    """
    results: list[CaseResult] = []
    for case in cases:
        agent = agent_factory()
        state = agent.run(case.question)

        cite_acc = evaluate_citation_accuracy(state)
        cov = evaluate_coverage(judge_llm, case, state)
        results.append(
            CaseResult(
                question=case.question,
                citation_accuracy=cite_acc,
                coverage=cov,
                total_tokens=agent.llm.usage.total_tokens,
                llm_calls=agent.llm.usage.calls,
                num_findings=len(state.all_findings()),
            )
        )

    _print_summary(results)
    return results


def _print_summary(results: list[CaseResult]) -> None:
    if not results:
        print("没有评估结果。")
        return
    n = len(results)
    avg_cite = sum(r.citation_accuracy for r in results) / n
    avg_cov = sum(r.coverage for r in results) / n
    avg_tok = sum(r.total_tokens for r in results) / n
    avg_calls = sum(r.llm_calls for r in results) / n

    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"用例数:              {n}")
    print(f"平均引用准确率:       {avg_cite:.1%}")
    print(f"平均覆盖度:           {avg_cov:.1%}")
    print(f"平均 token 消耗:      {avg_tok:.0f}")
    print(f"平均模型调用次数:     {avg_calls:.1f}")
    print("=" * 60)
