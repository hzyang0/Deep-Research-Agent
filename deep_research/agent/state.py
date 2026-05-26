"""
Agent 状态 (State)
==================

设计意图
--------
整个研究过程的"单一事实来源 (single source of truth)"。一个结构化的 state 对象
贯穿规划 → 搜索 → 反思 → 写作的全流程。

为什么用结构化 state 而不是堆叠对话历史(关键面试谈资)
--------------------------------------------------
很多入门级 Agent 把所有中间结果都拼进一个不断膨胀的 prompt / 对话历史里。问题:
- **上下文爆炸**:轮次一多,token 直线上升,既贵又触达上下文窗口上限。
- **不可控**:模型容易被早期无关内容干扰("迷失在中间"问题)。
- **难调试**:出了问题你很难定位是哪一步的数据坏了。

改用结构化 state 后:
- 每一步只读它需要的字段,上下文精确可控。
- state 可以直接打印 / 序列化,调试和可观测性大幅提升。
- `search_count` 这样的计数器天然支持防死循环逻辑。

这是从"会写 prompt"到"会做 Agent 工程"的认知分水岭。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..tools.fetch import Finding


@dataclass
class ResearchState:
    """一次研究任务的完整状态。"""

    # 用户的原始研究问题
    original_question: str

    # 规划器拆出的子问题列表
    sub_questions: list[str] = field(default_factory=list)

    # 每个子问题对应的发现(findings)。key 为子问题文本。
    findings: dict[str, list[Finding]] = field(default_factory=dict)

    # 每个子问题已经搜索的次数,用于防死循环(硬上限)。
    search_count: dict[str, int] = field(default_factory=dict)

    # 最终报告(写作阶段填充)
    report: str | None = None

    # 全程的轨迹日志,用于可观测性 / 调试 / 面试演示"它到底做了什么"。
    trace: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """记录一条轨迹。同时打印,方便实时观察 Agent 的"思考过程"。"""
        self.trace.append(message)
        print(message)

    def all_findings(self) -> list[Finding]:
        """把所有子问题的发现摊平成一个列表,供报告生成阶段使用。"""
        out: list[Finding] = []
        for items in self.findings.values():
            out.extend(items)
        return out
