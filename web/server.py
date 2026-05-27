"""
Web 后端 (FastAPI + WebSocket,多轮研究对话)
=============================================

设计意图
--------
把多轮研究 Agent 暴露成一个 WebSocket 聊天服务。这一层是表现层,很薄 ——
不含研究逻辑,只负责:连接管理、把前端消息转交给 ConversationSession、
把 Session 产生的实时事件推回浏览器。

为什么这次用 WebSocket 而非 SSE(关键面试谈资)
----------------------------------------------
需求从"单向看结果"升级为"多轮对话 + 实时追问/深入指令",数据流变成**双向**:
前端要在会话中持续向后端发消息,后端也要持续推事件。SSE 只能服务器→浏览器单向,
做双向得拼凑("SSE 收 + POST 发"),既别扭又难管理会话连接。WebSocket 是为
双向、长连接、有状态会话设计的,正好匹配。

这是一次有据可依的技术演进:之前单向需求时我用 SSE(够用就好),现在双向需求
明确了才升级到 WebSocket —— 选型由需求驱动,而不是追新。

并发模型
--------
Agent 的 run() 是同步阻塞的。WebSocket 处理协程里不能直接跑它(会阻塞事件循环)。
所以用 asyncio.to_thread 把整轮处理丢到线程池,事件通过 asyncio 队列桥接回
协程再 await send_json。这是"同步阻塞任务 + 异步双向通道"的标准桥接。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from deep_research.agent.conversation import ConversationSession
from deep_research.llm.client import LLMClient, LLMConfig

app = FastAPI(title="Deep Research Agent")
_STATIC_DIR = Path(__file__).parent / "static"
_DB_PATH = os.getenv("SESSIONS_DB", "sessions.db")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
        "smart_model": os.getenv("SMART_MODEL", "gpt-4o"),
        "fast_model": os.getenv("FAST_MODEL", "gpt-4o-mini"),
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """一条 WebSocket 连接 = 一段多轮研究会话。"""
    await ws.accept()

    # 每条连接持有一个会话。客户端可在首条消息里带 session_id 以恢复历史。
    session: ConversationSession | None = None

    try:
        while True:
            msg = await ws.receive_json()
            user_text = (msg.get("message") or "").strip()
            if not user_text:
                continue

            # 懒初始化会话(第一条消息时建立 / 恢复)
            if session is None:
                sid = msg.get("session_id")
                if sid:
                    session = await asyncio.to_thread(
                        ConversationSession.load, sid, _DB_PATH
                    )
                if session is None:
                    try:
                        llm = LLMClient(LLMConfig())
                    except Exception as err:  # noqa: BLE001
                        await ws.send_json({"type": "error", "message": f"LLM 初始化失败:{err}"})
                        continue
                    session = ConversationSession(llm=llm)
                await ws.send_json({"type": "session", "session_id": session.session_id})

            # 用 asyncio 队列把"线程里产生的事件"桥接回"协程里 send"
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def sink(event_type: str, data: dict) -> None:
                # 在工作线程里被调用,用 call_soon_threadsafe 安全投递到事件循环
                loop.call_soon_threadsafe(queue.put_nowait, {"type": event_type, **data})

            # 启动后台处理这一轮
            async def process():
                result = await asyncio.to_thread(session.handle_message, user_text, sink)
                await asyncio.to_thread(session.save, _DB_PATH)  # 持久化
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "turn_done", **result})

            task = asyncio.create_task(process())

            # 不断从队列取事件推给前端,直到本轮结束
            while True:
                event = await queue.get()
                await ws.send_json(event)
                if event.get("type") == "turn_done":
                    break
            await task

    except WebSocketDisconnect:
        # 连接断开,保存会话(若已建立)
        if session is not None:
            try:
                await asyncio.to_thread(session.save, _DB_PATH)
            except Exception:  # noqa: BLE001
                pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
