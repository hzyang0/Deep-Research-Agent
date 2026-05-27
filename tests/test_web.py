"""
Web 后端集成测试
================

验证 Web 层(FastAPI + SSE)能正确把 Agent 事件流式推给客户端 —— 通过 mock 掉
LLM 和工具,无需 API key、不联网即可运行。这是"表现层不含业务逻辑、只做转发"
这一设计的验证。运行:pytest tests/test_web.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import web.server as server  # noqa: E402
from deep_research.tools.fetch import Finding  # noqa: E402


class _FakeUsage:
    total_tokens = 999
    prompt_tokens = 800
    completion_tokens = 199
    calls = 5


class FakeLLM:
    def __init__(self, *a, **k):
        self.usage = _FakeUsage()

    def complete(self, prompt, **k):
        if "研究分析师" in prompt or "撰写" in prompt:
            return "## 结论\n关键论断 [1]。"
        if "搜索查询" in prompt:
            return "test query"
        return "ok"

    def complete_json(self, prompt, **k):
        if "拆解" in prompt:
            return ["子问题甲", "子问题乙"]
        if "是否已经足以" in prompt:
            return {"enough": True, "reason": "够了"}
        if "尚未覆盖" in prompt:
            return []
        return {}


def _patch(monkeypatch_targets):
    """把 server 里用到的 LLMClient 和抓取工具换成假的。"""
    server.LLMClient = FakeLLM  # type: ignore

    # 让 DeepResearchAgent 内部的抓取工具直接返回 Finding,不联网
    from deep_research.agent import agent as agent_mod

    orig_init = agent_mod.DeepResearchAgent.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)

        def fake_run(*, url, title, sub_question):
            return Finding(content="相关摘要", source_url=url, source_title=title)

        self.fetch_tool.run = fake_run

        class FakeSearch:
            def search(self, query, k=5):
                from deep_research.tools.search import SearchResult
                return [SearchResult(title="来源X", url="https://example.com/x", snippet="..")]

        self.search_tool = FakeSearch()

    agent_mod.DeepResearchAgent.__init__ = patched_init


def test_health_endpoint():
    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served():
    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "深度研究助手" in r.text


def test_sse_stream_emits_events():
    """核心测试:研究接口能流式吐出 plan / finding / done 等事件。"""
    _patch(None)
    from fastapi.testclient import TestClient
    client = TestClient(server.app)

    with client.stream(
        "POST", "/api/research",
        json={"question": "测试问题", "max_subquestions": 2, "max_searches_per_subq": 1},
    ) as resp:
        assert resp.status_code == 200
        types_seen = []
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = json.loads(line[5:].strip())
            types_seen.append(payload.get("type"))
            if payload.get("type") == "end":
                break

    # 验证关键事件都出现了
    assert "plan" in types_seen, "应推送规划事件"
    assert "finding" in types_seen, "应推送发现事件"
    assert "done" in types_seen, "应推送完成事件"
    assert "end" in types_seen, "应有结束哨兵"


if __name__ == "__main__":
    test_health_endpoint()
    test_index_served()
    test_sse_stream_emits_events()
    print("✅ Web 集成测试全部通过")
