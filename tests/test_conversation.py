"""
会话层与意图分类测试
====================
验证多轮会话的核心逻辑(意图路由、发现累积、持久化),全程 mock,不联网。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.agent.conversation import ConversationSession
from deep_research.agent.intent import Intent, classify_intent
from deep_research.tools.fetch import Finding
from deep_research.tools.search import SearchResult


class _Usage:
    total_tokens=10; prompt_tokens=8; completion_tokens=2; calls=1

class FakeLLM:
    def __init__(self): self.usage=_Usage()
    def complete(self, prompt, **k):
        if "研究分析师" in prompt or "撰写" in prompt: return "报告 [1]"
        if "搜索查询" in prompt: return "q"
        return "答案"
    def complete_json(self, prompt, **k):
        if "拆解" in prompt: return ["甲","乙"]
        if "是否已经足以" in prompt: return {"enough":True}
        if "尚未覆盖" in prompt: return []
        if "意图分类" in prompt: return {"intent":"FOLLOW_UP"}
        if "新子问题" in prompt: return ["新问题"]
        return {}

class FakeSearch:
    def search(self,q,k=5): return [SearchResult(title="T",url="https://e.com",snippet="")]


def test_intent_shortcircuit_no_history():
    """没有历史研究时,意图必然短路为 NEW_RESEARCH(省一次 LLM 调用)。"""
    assert classify_intent(FakeLLM(), "任意", has_prior_research=False, prior_topic=None) == Intent.NEW_RESEARCH


def _session():
    s = ConversationSession(llm=FakeLLM(), search_tool=FakeSearch())
    # mock 抓取
    from deep_research.agent import agent as am
    oi = am.DeepResearchAgent.__init__
    def ai(self,*a,**k):
        oi(self,*a,**k)
        self.fetch_tool.run = lambda *,url,title,sub_question: Finding(content="摘要",source_url=url,source_title=title)
        self.search_tool = FakeSearch()
    am.DeepResearchAgent.__init__ = ai
    return s


def test_new_research_then_followup_reuses_findings():
    """第一轮新研究累积发现;第二轮追问应复用发现、不清空。"""
    s = _session()
    r1 = s.handle_message("对比向量数据库")
    assert r1["is_report"] is True
    n_findings = len(s.findings)
    assert n_findings > 0

    r2 = s.handle_message("第二点再说说")  # FakeLLM 分类为 FOLLOW_UP
    assert r2["intent"] == "FOLLOW_UP"
    assert r2["is_report"] is False
    # 追问不应改变已有发现数量(没有新检索)
    assert len(s.findings) == n_findings


def test_persistence_roundtrip(tmp_path):
    """会话存盘后能原样恢复。"""
    s = _session()
    s.handle_message("对比向量数据库")
    db = str(tmp_path/"s.db")
    s.save(db)
    loaded = ConversationSession.load(s.session_id, db, llm=FakeLLM(), search_tool=FakeSearch())
    assert loaded is not None
    assert loaded.topic == s.topic
    assert len(loaded.findings) == len(s.findings)
    assert len(loaded.turns) == len(s.turns)


if __name__ == "__main__":
    import tempfile
    test_intent_shortcircuit_no_history()
    test_new_research_then_followup_reuses_findings()
    test_persistence_roundtrip(Path(tempfile.mkdtemp()))
    print("✅ 会话层测试通过")
