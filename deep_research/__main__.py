"""
命令行入口 (CLI Entry Point)
============================

用法::

    # 设置好环境变量后:
    export OPENAI_API_KEY=sk-...
    python -m deep_research "对比 2024-2025 年主流开源向量数据库的优劣"

    # 也可以调参:
    python -m deep_research "你的问题" --max-subq 4 --max-search 2

设计意图:提供一个零门槛的运行入口,方便演示和 eval。把所有可调参数暴露成
命令行参数,呼应"这些是我可以调的旋钮"这一面试叙事。
"""

from __future__ import annotations

import argparse
import sys

from .agent.agent import DeepResearchAgent
from .llm.client import LLMClient, LLMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="深度研究 Agent")
    parser.add_argument("question", help="要研究的问题")
    parser.add_argument("--max-subq", type=int, default=5, help="子问题数量上限")
    parser.add_argument("--max-search", type=int, default=3, help="每个子问题搜索次数上限")
    parser.add_argument("--top-k", type=int, default=4, help="每次搜索抓取的结果数")
    parser.add_argument("--output", type=str, default=None, help="把报告写入文件")
    args = parser.parse_args()

    try:
        llm = LLMClient(LLMConfig())
    except Exception as err:  # noqa: BLE001
        print(f"初始化 LLM 失败:{err}", file=sys.stderr)
        print("请确认已设置 OPENAI_API_KEY(以及可选的 OPENAI_BASE_URL)。", file=sys.stderr)
        sys.exit(1)

    agent = DeepResearchAgent(
        llm=llm,
        max_subquestions=args.max_subq,
        max_searches_per_subq=args.max_search,
        results_per_search=args.top_k,
    )
    state = agent.run(args.question)

    print("\n" + "=" * 60)
    print("最终报告")
    print("=" * 60)
    print(state.report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(state.report or "")
        print(f"\n报告已写入:{args.output}")


if __name__ == "__main__":
    main()
