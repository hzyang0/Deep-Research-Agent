"""
测试 (Tests)
============

设计意图
--------
这些测试**不需要 API key、不需要联网**就能跑通,通过 mock 掉 LLM 和搜索工具,
专门验证 Agent 的"控制流逻辑"是否正确 —— 比如:
- 防死循环的硬上限是否真的生效
- 引用越界是否被 eval 正确判为不准确
- 优雅降级(搜索返回空)是否不会让流程崩溃

为什么这很重要(面试谈资)
------------------------
Agent 系统的非确定性主要来自 LLM 和外部工具。把它们 mock 掉之后,我们就能
**确定性地**测试自己写的编排逻辑。这体现了"把不确定的部分隔离、对确定的部分
做严格测试"的工程思路。运行:`pytest tests/ -v`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.agent.agent import DeepResearchAgent
from deep_research.agent.state import ResearchState
from deep_research.eval.evaluator import evaluate_citation_accuracy
from deep_research.tools.fetch import Finding
from deep_research.tools.search import SearchResult, SearchTool


class FakeLLM:
    """假的 LLM:对不同提示返回预设答案,完全确定、零成本、不联网。"""

    def __init__(self):
        self.usage = _FakeUsage()

    def complete(self, prompt, *, tier=None, system=None, temperature=None, max_tokens=None):
        # 通过提示词里的关键词粗略判断当前是哪个步骤
        if "搜索查询" in prompt:
            return "test query"
        if "网页的正文" in prompt:
            return "这是一条针对子问题的相关摘要。"
        if "研究分析师" in prompt or "撰写" in prompt:
            # 生成一份带合法引用的报告
            return "## 结论\n这是一个关键论断 [1]。另一个论断 [2]。"
        return "ok"

    def complete_json(self, prompt, *, tier=None, system=None, temperature=None):
        if "拆解" in prompt:
            return ["子问题A", "子问题B"]
        if "是否已经足以" in prompt:
            return {"enough": True, "reason": "信息充分"}
        if "尚未覆盖" in prompt:
            return []  # 全局反思:无缺口
        return {}


class _FakeUsage:
    total_tokens = 1234
    prompt_tokens = 1000
    completion_tokens = 234
    calls = 7


class FakeSearchTool(SearchTool):
    """假的搜索工具:返回固定结果,不联网。"""

    def search(self, query, k=5):
        return [
            SearchResult(title="来源1", url="https://example.com/1", snippet="..."),
            SearchResult(title="来源2", url="https://example.com/2", snippet="..."),
        ]


def _build_agent(monkeypatch_fetch=True):
    agent = DeepResearchAgent(
        llm=FakeLLM(),
        search_tool=FakeSearchTool(),
        max_subquestions=2,
        max_searches_per_subq=2,
        results_per_search=2,
    )
    # 把抓取工具也 mock 掉:直接返回 Finding,不真正联网抓网页。
    def fake_run(*, url, title, sub_question):
        return Finding(content="相关摘要", source_url=url, source_title=title)

    agent.fetch_tool.run = fake_run  # type: ignore[assignment]
    return agent


def test_full_pipeline_runs():
    """端到端冒烟测试:整个控制流能跑通并产出报告。"""
    agent = _build_agent()
    state = agent.run("测试问题")
    assert state.report is not None
    assert len(state.sub_questions) == 2
    assert state.all_findings(), "应当收集到一些发现"


def test_dead_loop_guard():
    """防死循环:即使反思一直说"不够",也不会超过搜索次数上限。"""
    agent = _build_agent()

    # 让反思永远返回"不够",逼迫它一直想搜
    def never_enough(prompt, *, tier=None, system=None, temperature=None):
        if "是否已经足以" in prompt:
            return {"enough": False, "reason": "还不够"}
        if "拆解" in prompt:
            return ["唯一子问题"]
        if "尚未覆盖" in prompt:
            return []
        return {}

    agent.llm.complete_json = never_enough  # type: ignore[assignment]
    state = agent.run("测试问题")
    # 关键断言:搜索次数被硬上限卡住,不会无限循环
    for subq in state.sub_questions:
        assert state.search_count[subq] <= agent.max_searches_per_subq


def test_graceful_degradation_on_empty_search():
    """优雅降级:搜索返回空结果时,流程不崩溃,仍产出报告。"""
    agent = _build_agent()

    class EmptySearch(SearchTool):
        def search(self, query, k=5):
            return []

    agent.search_tool = EmptySearch()
    state = agent.run("测试问题")
    assert state.report is not None  # 没崩,而是优雅地走到报告阶段


def test_citation_accuracy_detects_hallucinated_index():
    """eval 校验:越界引用 [99] 应被判为不准确。"""
    state = ResearchState(original_question="q")
    state.findings["sub"] = [Finding(content="c", source_url="u", source_title="t")]
    # 只有 1 条发现,合法编号只有 [1];报告里却引用了 [1] 和越界的 [99]
    state.report = "论断一 [1]。论断二 [99]。\n\n## 参考来源\n[1] t - u"
    acc = evaluate_citation_accuracy(state)
    assert acc == 0.5, f"期望 0.5(2 个引用 1 个合法),实际 {acc}"


if __name__ == "__main__":
    # 允许直接 python tests/test_agent.py 运行(不依赖 pytest)
    test_full_pipeline_runs()
    test_dead_loop_guard()
    test_graceful_degradation_on_empty_search()
    test_citation_accuracy_detects_hallucinated_index()
    print("✅ 所有测试通过")
