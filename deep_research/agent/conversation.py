"""
研究会话 (Conversation Session)
===============================

设计意图
--------
把"一次性研究"升级为"有状态的多轮研究对话"。一个 ConversationSession 代表
用户和 Agent 的一整段对话,它:
1. 累积所有轮次产生的研究发现(跨轮复用,这是"多轮"的价值所在);
2. 维护对话历史;
3. 根据意图分类,把每轮用户消息路由到不同处理逻辑;
4. 通过 SQLite 持久化,会话可恢复。

为什么单独做一层,而不是把多轮逻辑塞进 DeepResearchAgent(面试谈资)
----------------------------------------------------------------
关注点分离。DeepResearchAgent 的职责是"把一个问题研究透"(单轮、无状态)。
多轮对话的状态管理、意图路由、历史维护是另一组关注点。把它们分到 Session 层,
让 DeepResearchAgent 保持简单、可独立测试、可被 CLI 直接复用。Session 是
"编排 Agent 的更上层编排器"。

意图路由(本层的核心价值)
------------------------
- NEW_RESEARCH → 调用 DeepResearchAgent 跑完整研究,结果并入会话。
- DEEPEN       → 针对指定方面追加子问题并定向检索,发现并入会话。
- FOLLOW_UP    → 不检索,直接用已累积发现回答(省钱省时)。
- REFINE       → 追加新角度的子问题并检索。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..llm.client import LLMClient
from ..tools.fetch import Finding
from ..tools.search import SearchTool
from . import steps
from .agent import DeepResearchAgent
from .intent import Intent, answer_follow_up, classify_intent
from .state import ResearchState


@dataclass
class Turn:
    """对话中的一轮。role 为 'user' 或 'assistant'。"""

    role: str
    content: str
    intent: str | None = None  # 用户轮记录被分类成的意图,便于调试/展示
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationSession:
    """一段多轮研究对话的状态容器与编排器。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        search_tool: SearchTool | None = None,
        *,
        session_id: str | None = None,
        max_searches_per_subq: int = 2,
        results_per_search: int = 4,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.llm = llm or LLMClient()
        self.search_tool = search_tool or SearchTool()
        self.max_searches_per_subq = max_searches_per_subq
        self.results_per_search = results_per_search

        # 会话累积状态
        self.turns: list[Turn] = []
        self.topic: str | None = None          # 当前研究主题(第一轮研究问题)
        self.sub_questions: list[str] = []      # 累积的所有子问题
        self.findings: list[Finding] = []       # 累积的所有发现(跨轮复用的核心)
        self.latest_report: str | None = None

    # ---------- 对外主入口 ----------

    def handle_message(self, user_message: str, event_sink=None) -> dict:
        """处理用户一轮消息,返回结构化结果(供 Web 层推给前端)。

        event_sink 用于在处理过程中实时推送 Agent 事件(规划/检索/发现…)。
        """
        self.turns.append(Turn(role="user", content=user_message))

        # 1. 意图分类(没有历史研究时会规则短路为 NEW_RESEARCH)
        intent = classify_intent(
            self.llm,
            user_message,
            has_prior_research=bool(self.findings),
            prior_topic=self.topic,
            sub_questions=self.sub_questions,
        )
        self.turns[-1].intent = intent.value
        if event_sink:
            event_sink("intent", {"intent": intent.value})

        # 2. 路由到对应处理
        if intent == Intent.NEW_RESEARCH:
            result = self._do_new_research(user_message, event_sink)
        elif intent == Intent.FOLLOW_UP:
            result = self._do_follow_up(user_message, event_sink)
        else:  # DEEPEN / REFINE 都走"追加子问题并检索",只是 prompt 略不同
            result = self._do_deepen_or_refine(user_message, intent, event_sink)

        # 3. 记录助手轮
        self.turns.append(Turn(role="assistant", content=result.get("answer", "")))
        return result

    # ---------- 四种意图的处理 ----------

    def _do_new_research(self, question: str, event_sink) -> dict:
        """全新研究:跑完整 Agent 流程,结果并入会话。"""
        agent = DeepResearchAgent(
            llm=self.llm,
            search_tool=self.search_tool,
            max_searches_per_subq=self.max_searches_per_subq,
            results_per_search=self.results_per_search,
        )
        state = agent.run(question, event_sink=event_sink)

        # 并入会话累积状态
        self.topic = question
        self.sub_questions = list(state.sub_questions)
        self.findings = state.all_findings()
        self.latest_report = state.report
        return {
            "intent": Intent.NEW_RESEARCH.value,
            "answer": state.report or "",
            "is_report": True,
            "sources": self._sources_payload(),
        }

    def _do_follow_up(self, user_message: str, event_sink) -> dict:
        """追问:不检索,直接用已累积发现回答。"""
        if event_sink:
            event_sink("phase", {"name": "answering", "label": "基于已有发现作答(无需检索)"})
        context = self._findings_as_text()
        answer = answer_follow_up(self.llm, user_message, context)
        return {
            "intent": Intent.FOLLOW_UP.value,
            "answer": answer,
            "is_report": False,
            "sources": self._sources_payload(),
        }

    def _do_deepen_or_refine(self, user_message: str, intent: Intent, event_sink) -> dict:
        """深入/补充:把用户诉求转成 1-2 个新子问题,定向检索,发现并入会话,
        然后给出针对性回答。"""
        if event_sink:
            label = "深入检索" if intent == Intent.DEEPEN else "补充新角度检索"
            event_sink("phase", {"name": "researching", "label": label})

        # 用 LLM 把用户的自然语言诉求转成具体可检索的子问题
        new_subqs = self._derive_subquestions(user_message, intent)
        if event_sink:
            event_sink("plan", {"sub_questions": new_subqs})

        # 复用 DeepResearchAgent 的内循环来检索这些新子问题
        agent = DeepResearchAgent(
            llm=self.llm,
            search_tool=self.search_tool,
            max_searches_per_subq=self.max_searches_per_subq,
            results_per_search=self.results_per_search,
        )
        # 借用一个临时 state 承载本轮检索(共享 event_sink 以便前端实时看到)
        tmp = ResearchState(original_question=user_message, event_sink=event_sink)
        for sq in new_subqs:
            tmp.findings[sq] = []
            tmp.search_count[sq] = 0
            agent._research_subquestion(tmp, sq)  # 复用内循环,不重复造轮子

        new_findings = tmp.all_findings()
        # 并入会话累积状态
        self.sub_questions.extend(new_subqs)
        self.findings.extend(new_findings)

        if event_sink:
            event_sink("phase", {"name": "writing", "label": "综合新旧发现作答"})
        # 基于"新发现 + 已有上下文"作答
        answer = answer_follow_up(self.llm, user_message, self._findings_as_text())
        return {
            "intent": intent.value,
            "answer": answer,
            "is_report": False,
            "sources": self._sources_payload(),
        }

    # ---------- 辅助 ----------

    def _derive_subquestions(self, user_message: str, intent: Intent) -> list[str]:
        """把用户的深入/补充诉求转成 1-2 个具体子问题。"""
        hint = (
            "用户想就已有研究的某方面深入了解更多细节"
            if intent == Intent.DEEPEN
            else "用户想补充一个之前没覆盖的新角度"
        )
        prompt = (
            f"当前研究主题:{self.topic}\n"
            f"已有子问题:{self.sub_questions}\n"
            f"{hint}。用户说:「{user_message}」\n\n"
            f"请把用户的诉求转化成 1-2 个具体、可通过网络搜索回答的新子问题。"
            f'以 JSON 数组返回,例如 ["子问题1"]。'
        )
        from ..llm.client import ModelTier

        data = self.llm.complete_json(prompt, tier=ModelTier.SMART, temperature=0.0)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        if not isinstance(data, list):
            return [user_message]  # 降级:直接拿用户原话当子问题
        return [str(x) for x in data][:2]

    def _findings_as_text(self) -> str:
        return "\n".join(
            f"[{i+1}] {f.content}(来源:{f.source_title})"
            for i, f in enumerate(self.findings)
        )

    def _sources_payload(self) -> list[dict]:
        return [
            {"title": f.source_title, "url": f.source_url} for f in self.findings
        ]

    # ---------- SQLite 持久化 ----------

    def save(self, db_path: str = "sessions.db") -> None:
        """把会话存进 SQLite。发现/历史序列化为 JSON 存字段里。

        持久化让会话可恢复,也为多用户/多会话留好扩展空间。生产环境可换成
        正式数据库,这里用 SQLite 是因为零配置、单文件、Python 内置。
        """
        conn = _connect(db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, topic, data, updated_at) VALUES (?,?,?,?)",
                (
                    self.session_id,
                    self.topic,
                    json.dumps(self._serialize(), ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def load(cls, session_id: str, db_path: str = "sessions.db", **kwargs) -> "ConversationSession | None":
        conn = _connect(db_path)
        try:
            row = conn.execute(
                "SELECT data FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        sess = cls(session_id=session_id, **kwargs)
        sess._deserialize(json.loads(row[0]))
        return sess

    def _serialize(self) -> dict:
        return {
            "topic": self.topic,
            "sub_questions": self.sub_questions,
            "findings": [
                {"content": f.content, "source_url": f.source_url, "source_title": f.source_title}
                for f in self.findings
            ],
            "latest_report": self.latest_report,
            "turns": [
                {"role": t.role, "content": t.content, "intent": t.intent, "timestamp": t.timestamp}
                for t in self.turns
            ],
        }

    def _deserialize(self, d: dict) -> None:
        self.topic = d.get("topic")
        self.sub_questions = d.get("sub_questions", [])
        self.findings = [
            Finding(content=f["content"], source_url=f["source_url"], source_title=f["source_title"])
            for f in d.get("findings", [])
        ]
        self.latest_report = d.get("latest_report")
        self.turns = [
            Turn(role=t["role"], content=t["content"], intent=t.get("intent"), timestamp=t.get("timestamp", ""))
            for t in d.get("turns", [])
        ]


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "id TEXT PRIMARY KEY, topic TEXT, data TEXT, updated_at TEXT)"
    )
    return conn
