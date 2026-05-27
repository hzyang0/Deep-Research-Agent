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
from typing import Any, Callable

from ..tools.fetch import Finding

# 事件回调的类型:接收 (事件类型, 数据字典)。
# 设计意图:这是连接"核心 Agent"和"任意前端(CLI/Web)"的唯一桥梁。
# Agent 内部只管发事件,完全不知道接收方是终端还是浏览器 —— 这就是
# 表现层(presentation)与业务逻辑(domain)解耦。面试谈资:
# "我的 Agent 不依赖任何特定 UI,CLI 和 Web 共用同一套核心,只是事件的消费者不同。"
EventSink = Callable[[str, dict[str, Any]], None]


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

    # 可选的事件回调。为 None 时(默认,CLI 场景)行为和原来完全一致;
    # Web 场景会注入一个把事件转成 SSE 推给浏览器的回调。
    event_sink: EventSink | None = None

    def log(self, message: str) -> None:
        """记录一条轨迹。同时打印,方便实时观察 Agent 的"思考过程"。"""
        self.trace.append(message)
        print(message)
        # 把纯文本日志也作为一种事件发出去,Web 端可直接展示。
        self.emit("log", {"message": message})

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """发出一个结构化事件(如果注册了 sink)。

        相比纯文本 log,结构化事件让前端能精细渲染:比如 "plan" 事件带着
        子问题列表,前端就能画成一组卡片,而不只是打印一行字。
        失败时静默吞掉异常 —— 事件推送绝不能影响核心研究流程(降级原则)。
        """
        if self.event_sink is None:
            return
        try:
            self.event_sink(event_type, data)
        except Exception:  # noqa: BLE001 —— UI 推送失败不应中断研究
            pass

    def all_findings(self) -> list[Finding]:
        """把所有子问题的发现摊平成一个列表,供报告生成阶段使用。"""
        out: list[Finding] = []
        for items in self.findings.values():
            out.extend(items)
        return out
