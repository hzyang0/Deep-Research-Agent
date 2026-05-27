"""
深度研究 Agent 主控制循环 (Agent Orchestration Loop)
====================================================

这是整个项目的灵魂。它把前面所有模块(LLM、工具、推理步骤、状态)编排成一个
完整的"规划 → 执行 → 反思"循环。

控制流(就是你面试时要手画的那张图)
----------------------------------
    开始
     │
     ▼
   [规划器] 把研究问题拆成若干子问题
     │
     ▼
   对每个子问题:
     ├─ while 未达搜索次数上限:
     │    ├─ [改写查询] 基于已有发现生成搜索词
     │    ├─ [搜索]      拿到候选网页
     │    ├─ [抓取+摘要]  逐个提取相关发现(小模型)
     │    └─ [反思]      信息够了吗?够→break,不够→继续
     │
     ▼
   [全局反思] 还有缺口吗?有→补充子问题再搜
     │
     ▼
   [报告生成] 综合所有发现,产出带引用的报告(大模型)
     │
     ▼
    结束

防死循环的两道防线(面试高频追问)
--------------------------------
1. **每个子问题的搜索次数硬上限** (`max_searches_per_subq`):
   即使反思器一直说"不够",也不会无限搜下去。
2. **全局补充子问题的数量上限**:全局反思最多补 2 个,且补充的子问题同样受
   单子问题搜索上限约束。
这两道防线保证流程一定会终止,且总成本有上界 —— 这是工程纪律,不是可有可无。
"""

from __future__ import annotations

from ..llm.client import LLMClient
from ..tools.fetch import FetchSummarizeTool
from ..tools.search import SearchTool
from . import steps
from .state import ResearchState


class DeepResearchAgent:
    """深度研究 Agent。

    用法::

        agent = DeepResearchAgent()
        state = agent.run("对比 2024-2025 主流开源向量数据库的优劣")
        print(state.report)
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        search_tool: SearchTool | None = None,
        *,
        max_subquestions: int = 5,
        max_searches_per_subq: int = 3,
        results_per_search: int = 4,
    ) -> None:
        self.llm = llm or LLMClient()
        self.search_tool = search_tool or SearchTool()
        self.fetch_tool = FetchSummarizeTool(self.llm)

        # —— 这些就是面试时要讲的"可调参数(旋钮)" ——
        # 子问题数量上限:太多则成本爆炸,太少则覆盖不全。
        self.max_subquestions = max_subquestions
        # 每个子问题最多搜索几次:成本与质量的核心权衡,也是防死循环的硬上限。
        self.max_searches_per_subq = max_searches_per_subq
        # 每次搜索抓取前 N 条结果:top-k 的选择,影响成本与召回。
        self.results_per_search = results_per_search

    def run(self, question: str, event_sink=None) -> ResearchState:
        """执行一次完整的深度研究,返回包含报告与轨迹的 state。

        event_sink: 可选的事件回调。CLI 调用时不传(行为不变);Web 调用时传入
        一个把事件转成 SSE 的回调,从而实现"实时看到 Agent 思考过程"。
        """
        state = ResearchState(original_question=question, event_sink=event_sink)
        state.emit("start", {"question": question})
        state.log(f"🎯 研究问题:{question}")

        # ---------- 阶段 1:规划 ----------
        state.emit("phase", {"name": "planning", "label": "正在规划子问题"})
        state.sub_questions = steps.plan_subquestions(
            self.llm, question, max_subq=self.max_subquestions
        )
        state.log(f"📋 规划出 {len(state.sub_questions)} 个子问题:")
        for sq in state.sub_questions:
            state.log(f"   - {sq}")
            state.findings[sq] = []
            state.search_count[sq] = 0
        # 结构化的规划事件:前端可据此画出一组子问题卡片
        state.emit("plan", {"sub_questions": list(state.sub_questions)})

        # ---------- 阶段 2:逐子问题执行(搜索-抓取-反思循环) ----------
        state.emit("phase", {"name": "researching", "label": "正在逐个研究子问题"})
        for subq in list(state.sub_questions):
            self._research_subquestion(state, subq)

        # ---------- 阶段 3:全局反思 + 补缺 ----------
        state.emit("phase", {"name": "reflecting", "label": "全局反思,检查覆盖缺口"})
        gaps = steps.reflect_global_gaps(self.llm, state)
        if gaps:
            state.log(f"🔍 全局反思发现 {len(gaps)} 个缺口,补充研究:")
            state.emit("gaps", {"gaps": gaps})
            for g in gaps:
                state.log(f"   + {g}")
                state.sub_questions.append(g)
                state.findings[g] = []
                state.search_count[g] = 0
                self._research_subquestion(state, g)
        else:
            state.log("✅ 全局反思:覆盖充分,无需补充。")

        # ---------- 阶段 4:生成报告 ----------
        state.emit("phase", {"name": "writing", "label": "正在生成带引用的报告"})
        state.log("📝 正在综合所有发现生成报告...")
        state.report = steps.generate_report(self.llm, state)

        # 收尾:打印成本(可观测性 —— 面试时最有说服力的数字)
        u = self.llm.usage
        state.log(
            f"💰 本次研究共调用模型 {u.calls} 次,"
            f"消耗 token:{u.total_tokens}"
            f"(prompt {u.prompt_tokens} + completion {u.completion_tokens})"
        )
        # 完成事件:把报告、来源、成本一次性结构化推给前端
        state.emit(
            "done",
            {
                "report": state.report,
                "sources": [
                    {"title": f.source_title, "url": f.source_url}
                    for f in state.all_findings()
                ],
                "usage": {
                    "calls": u.calls,
                    "total_tokens": u.total_tokens,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                },
            },
        )
        return state

    def _research_subquestion(self, state: ResearchState, subq: str) -> None:
        """针对单个子问题执行"搜索-抓取-反思"内循环。

        这是 Agent 内层循环,体现了"观察→思考→行动→再观察"的 agentic 本质。
        """
        state.log(f"\n🔬 开始研究子问题:{subq}")
        state.emit("subq_start", {"sub_question": subq})
        while state.search_count[subq] < self.max_searches_per_subq:
            round_no = state.search_count[subq] + 1

            # 行动 1:改写查询(基于已有发现)
            query = steps.refine_query(self.llm, subq, state.findings[subq])
            state.log(f"   [第 {round_no} 轮] 搜索查询:{query}")
            state.emit("search", {"sub_question": subq, "round": round_no, "query": query})

            # 行动 2:搜索
            results = self.search_tool.search(query, k=self.results_per_search)
            state.search_count[subq] += 1
            if not results:
                state.log("   ⚠️ 本轮搜索无结果(可能被限流),提前结束该子问题。")
                state.emit("search_empty", {"sub_question": subq})
                break

            # 行动 3:逐个抓取并定向摘要
            for r in results:
                finding = self.fetch_tool.run(
                    url=r.url, title=r.title, sub_question=subq
                )
                if finding:
                    state.findings[subq].append(finding)
                    state.log(f"      ✓ 收录:{finding.source_title[:40]}")
                    state.emit(
                        "finding",
                        {
                            "sub_question": subq,
                            "title": finding.source_title,
                            "url": finding.source_url,
                            "content": finding.content,
                        },
                    )

            # 思考:反思信息是否充分(终止条件)
            if steps.reflect_subquestion(self.llm, subq, state.findings[subq]):
                state.log(f"   ✅ 该子问题信息已充分(共 {len(state.findings[subq])} 条)。")
                state.emit(
                    "subq_done",
                    {"sub_question": subq, "reason": "enough", "count": len(state.findings[subq])},
                )
                break
        else:
            # while 正常跑完(达到次数上限)才会进入 else
            state.log(
                f"   ⏹️ 达到搜索次数上限 ({self.max_searches_per_subq}),"
                f"停止该子问题(共 {len(state.findings[subq])} 条发现)。"
            )
            state.emit(
                "subq_done",
                {"sub_question": subq, "reason": "max_search", "count": len(state.findings[subq])},
            )
