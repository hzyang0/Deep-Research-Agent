"""
Web 后端 (FastAPI + SSE)
========================

设计意图
--------
把命令行的深度研究 Agent 包装成一个 Web 服务,让用户能在浏览器里实时看到
Agent 的"思考过程"。这一层是**表现层(presentation)**,刻意做得很薄 ——
它不包含任何研究逻辑,只负责:
1. 接收前端的研究请求;
2. 在后台线程跑 Agent,把 Agent 发出的结构化事件通过 SSE 实时推给浏览器;
3. 托管前端静态文件。

为什么用 SSE 而不是 WebSocket(面试谈资)
--------------------------------------
研究过程的数据流是**单向**的:后端不断把进度(规划→搜索→反思→报告)推给前端,
前端不需要在研究过程中往回发消息。这正是 Server-Sent Events 的设计场景。
SSE 基于普通 HTTP,实现简单、自动重连、无需额外协议握手。WebSocket 是双向的,
功能更强但更重 —— 在"单向推送"场景下用它属于过度设计。
"选择恰好够用的技术,而不是最酷的技术",本身就是工程成熟度的体现。

为什么 Agent 跑在后台线程 + 队列(面试谈资)
----------------------------------------
Agent 的 run() 是同步阻塞的(里面是串行的网络请求)。如果直接在请求处理函数里跑,
会阻塞事件循环。所以我们把 Agent 丢到后台线程跑,它产生的事件投递到一个线程安全
队列,SSE 生成器从队列里取事件往外发。这是"同步阻塞任务 + 异步流式输出"的经典桥接。
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from deep_research.agent.agent import DeepResearchAgent
from deep_research.llm.client import LLMClient, LLMConfig

app = FastAPI(title="Deep Research Agent")

_STATIC_DIR = Path(__file__).parent / "static"


class ResearchRequest(BaseModel):
    """前端发来的研究请求。参数对应 Agent 的可调旋钮。"""

    question: str
    max_subquestions: int = 4
    max_searches_per_subq: int = 2
    results_per_search: int = 4


def _run_agent_to_queue(req: ResearchRequest, q: "queue.Queue") -> None:
    """在后台线程里跑 Agent,把每个事件投递到队列。

    用一个特殊的 None 哨兵标记"研究结束",让 SSE 生成器知道何时收尾。
    任何异常也转成一个 error 事件发出去,保证前端不会无限等待。
    """
    def sink(event_type: str, data: dict) -> None:
        q.put((event_type, data))

    try:
        llm = LLMClient(LLMConfig())
        agent = DeepResearchAgent(
            llm=llm,
            max_subquestions=req.max_subquestions,
            max_searches_per_subq=req.max_searches_per_subq,
            results_per_search=req.results_per_search,
        )
        agent.run(req.question, event_sink=sink)
    except Exception as err:  # noqa: BLE001
        q.put(("error", {"message": str(err)}))
    finally:
        q.put(None)  # 结束哨兵


@app.post("/api/research")
def research(req: ResearchRequest) -> StreamingResponse:
    """启动一次研究,以 SSE 流式返回全过程事件。"""
    q: "queue.Queue" = queue.Queue()
    worker = threading.Thread(target=_run_agent_to_queue, args=(req, q), daemon=True)
    worker.start()

    def event_stream():
        # SSE 格式:每条消息形如 "data: {json}\n\n"
        while True:
            item = q.get()
            if item is None:  # 结束哨兵
                yield "data: {\"type\": \"end\"}\n\n"
                break
            event_type, data = item
            payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲,保证实时性
        },
    )


@app.get("/api/health")
def health() -> dict:
    """健康检查 + 报告 LLM 是否已配置好(前端据此提示用户)。"""
    return {
        "status": "ok",
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
        "smart_model": os.getenv("SMART_MODEL", "gpt-4o"),
        "fast_model": os.getenv("FAST_MODEL", "gpt-4o-mini"),
    }


@app.get("/")
def index() -> FileResponse:
    """返回前端单页。"""
    return FileResponse(_STATIC_DIR / "index.html")


# 托管其它静态资源(目前前端是单文件,这一行为将来拆分留好余地)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
