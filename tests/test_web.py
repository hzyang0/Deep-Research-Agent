"""
Web 后端集成测试 (WebSocket 多轮对话)
=====================================

通过 mock 掉 LLM 和工具,无需 API key、不联网即可验证:
- WebSocket 能建立会话、流式推送 Agent 事件
- 多轮对话:第一轮新研究,第二轮追问(意图分类生效)
运行:pytest tests/test_web.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import web.server as server  # noqa: E402
from deep_research.tools.fetch import Finding  # noqa: E402
from deep_research.tools.search import SearchResult  # noqa: E402


class _Usage:
    total_tokens = 100; prompt_tokens = 80; completion_tokens = 20; calls = 3


class FakeLLM:
    def __init__(self, *a, **k): self.usage = _Usage()
    def complete(self, prompt, **k):
        if "研究分析师" in prompt or "撰写" in prompt: return "## 报告\n论断 [1]。"
        if "搜索查询" in prompt: return "q"
        if "追问" in prompt or "回答用户" in prompt: return "这是基于已有发现的追问回答。"
        return "ok"
    def complete_json(self, prompt, **k):
        if "拆解" in prompt: return ["子问题甲", "子问题乙"]
        if "是否已经足以" in prompt: return {"enough": True}
        if "尚未覆盖" in prompt: return []
        if "意图分类" in prompt: return {"intent": "FOLLOW_UP", "reason": "追问"}
        if "新子问题" in prompt: return ["新子问题"]
        return {}


class FakeSearch:
    def search(self, query, k=5):
        return [SearchResult(title="来源A", url="https://example.com/a", snippet="..")]


def _patch():
    server.LLMClient = FakeLLM  # type: ignore
    from deep_research.agent import conversation as conv
    conv.LLMClient = FakeLLM  # type: ignore

    # 让会话用假的搜索 + 假的抓取
    orig_init = conv.ConversationSession.__init__
    def patched(self, *a, **k):
        k.setdefault("search_tool", FakeSearch())
        orig_init(self, *a, **k)
        # 替换 agent 内部抓取
        from deep_research.agent import agent as am
        oi = am.DeepResearchAgent.__init__
        def ai(s, *aa, **kk):
            oi(s, *aa, **kk)
            s.fetch_tool.run = lambda *, url, title, sub_question: Finding(
                content="摘要", source_url=url, source_title=title)
            s.search_tool = FakeSearch()
        am.DeepResearchAgent.__init__ = ai
    conv.ConversationSession.__init__ = patched


def test_health():
    from fastapi.testclient import TestClient
    assert TestClient(server.app).get("/api/health").json()["status"] == "ok"


def test_index_served():
    from fastapi.testclient import TestClient
    r = TestClient(server.app).get("/")
    assert r.status_code == 200 and "深度研究助手" in r.text


def test_ws_multi_turn(tmp_path):
    """核心:WebSocket 多轮对话。第一轮新研究出报告,第二轮追问。"""
    _patch()
    server._DB_PATH = str(tmp_path / "t.db")
    from fastapi.testclient import TestClient
    client = TestClient(server.app)

    with client.websocket_connect("/ws") as ws:
        # 第一轮:新研究
        ws.send_json({"message": "对比向量数据库"})
        types1 = []
        while True:
            ev = ws.receive_json()
            types1.append(ev["type"])
            if ev["type"] == "turn_done":
                assert ev["is_report"] is True  # 新研究产出报告
                break
        assert "session" in types1
        assert "plan" in types1
        assert "finding" in types1

        # 第二轮:追问(应被分类为 FOLLOW_UP,不重新检索)
        ws.send_json({"message": "第二点能再解释下吗"})
        types2 = []
        while True:
            ev = ws.receive_json()
            types2.append(ev["type"])
            if ev["type"] == "turn_done":
                assert ev["intent"] == "FOLLOW_UP"
                assert ev["is_report"] is False  # 追问不是报告
                break
        # 追问不应触发 plan(没有新检索)
        assert "plan" not in types2


if __name__ == "__main__":
    import tempfile
    test_health(); test_index_served()
    test_ws_multi_turn(Path(tempfile.mkdtemp()))
    print("✅ Web 多轮对话测试通过")
