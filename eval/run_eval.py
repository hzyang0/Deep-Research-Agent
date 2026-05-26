"""
评估运行脚本
============

用法::

    export OPENAI_API_KEY=sk-...
    python eval/run_eval.py
    python eval/run_eval.py --max-search 2 --max-subq 3   # 对比不同参数下的指标

这个脚本是你做"迭代优化叙事"的工具:
1. 跑一次记下基线指标。
2. 改一处(prompt / 参数 / 摘要策略)。
3. 再跑一次,对比指标变化。
4. 把每次的指标和你的思考记进 docs/EXPERIMENTS.md。
这些记录就是你面试时"我把引用准确率从 X% 提到 Y%"故事的弹药。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 让脚本能找到 deep_research 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.agent.agent import DeepResearchAgent  # noqa: E402
from deep_research.eval.evaluator import EvalCase, run_eval  # noqa: E402
from deep_research.llm.client import LLMClient, LLMConfig  # noqa: E402


def load_cases() -> list[EvalCase]:
    path = Path(__file__).parent / "dataset" / "research_questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(question=d["question"], expected_points=d["expected_points"]) for d in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="对深度研究 Agent 跑评估")
    parser.add_argument("--max-subq", type=int, default=4)
    parser.add_argument("--max-search", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个用例(省钱调试)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY。", file=sys.stderr)
        sys.exit(1)

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    # 裁判用独立的 LLM 客户端,避免它的 token 用量混入被测 agent 的成本统计。
    judge = LLMClient(LLMConfig())

    def agent_factory() -> DeepResearchAgent:
        # 每个用例都新建 agent + 新的 LLMClient,保证 token 计数互相独立。
        return DeepResearchAgent(
            llm=LLMClient(LLMConfig()),
            max_subquestions=args.max_subq,
            max_searches_per_subq=args.max_search,
            results_per_search=args.top_k,
        )

    run_eval(agent_factory, cases, judge)


if __name__ == "__main__":
    main()
