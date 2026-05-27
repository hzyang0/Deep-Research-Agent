"""
LLM 客户端层 (LLM Client Layer)
================================

设计意图
--------
这一层是整个 Agent 的"地基"。它把"如何调用大模型"这件事和"Agent 的业务逻辑"
彻底解耦。Agent 的其它模块永远只调用 `LLMClient.complete()` 或 `.complete_json()`,
而不需要关心底层用的是 OpenAI、Claude、还是本地模型。

为什么这么设计(面试谈资)
------------------------
1. **模型分级 (model tiering)**:Agent 里不同步骤对模型能力的需求差异巨大。
   - 网页摘要这种"读一段文字、抽取要点"的任务 → 用便宜的小模型 (FAST tier)
   - 最终报告生成这种"综合材料、保证质量"的任务 → 用强一点的模型 (SMART tier)
   把这个区分放在基础设施层,业务代码里只需要声明 `tier=ModelTier.FAST`,
   成本优化就自然落地了。这是从"玩具 demo"走向"工程"的关键一步。

2. **结构化输出 (structured output)**:Agent 内部大量步骤(规划、反思)需要模型
   返回机器可解析的结果,而不是一段散文。`complete_json()` 封装了
   "要求模型只输出 JSON + 容错解析" 的逻辑,避免每个调用点都重复造轮子。

3. **可观测性 (observability)**:每次调用都累加 token 计数,这样我们能在最后
   报告这次研究花了多少 token —— 成本数字是面试时最有说服力的证据。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# 我们依赖 openai 这个 SDK,因为它的接口已经成为事实标准。
# 注意:通过设置 base_url,同样的代码可以指向任何"OpenAI 兼容"的服务,
# 包括 Claude(通过兼容层)、本地 Ollama、各类国产模型 API 等。
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # 允许在没装 SDK 的环境下仅做静态检查

# 模块导入时即加载 .env,确保后续 os.getenv 能读到用户配置。
# CLI 和 Web 都会 import 到本模块,所以这一处加载就覆盖了所有入口。
from ..config import load_env

load_env()


class ModelTier(str, Enum):
    """模型分级。

    FAST  -> 便宜、快,用于高频低难度任务(摘要、相关性判断)。
    SMART -> 较贵、较强,用于低频高难度任务(规划、反思、最终写作)。

    把"档位"而不是"具体模型名"暴露给业务层,是为了让"换模型"这件事
    只发生在配置里,不污染业务逻辑。
    """

    FAST = "fast"
    SMART = "smart"


@dataclass
class LLMUsage:
    """累计 token 用量,用于成本核算与可观测性。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1


@dataclass
class LLMConfig:
    """LLM 配置。所有可调项集中在这里,方便面试时讲"我把哪些旋钮暴露出来了"。"""

    # 两个档位分别对应的模型名。默认值用 OpenAI 的型号,换平台时改这里即可。
    fast_model: str = field(default_factory=lambda: os.getenv("FAST_MODEL", "gpt-4o-mini"))
    smart_model: str = field(default_factory=lambda: os.getenv("SMART_MODEL", "gpt-4o"))
    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    # 默认温度:规划/反思类调用会单独传入更低的温度以保证稳定。
    default_temperature: float = 0.2
    # 单次调用最大输出 token,防止失控的长输出烧钱。
    max_tokens: int = 2000
    # 调用失败时的重试次数(网络抖动、限流等)。
    max_retries: int = 2


class LLMClient:
    """对大模型调用的统一封装。

    用法::

        client = LLMClient()
        text = client.complete("你好", tier=ModelTier.FAST)
        data = client.complete_json("返回一个 JSON 列表", tier=ModelTier.SMART)
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.usage = LLMUsage()
        if OpenAI is None:
            raise RuntimeError(
                "未安装 openai SDK。请先 `pip install openai`,或在 mock 模式下运行测试。"
            )
        # base_url 为 None 时使用官方默认地址;非 None 时指向兼容服务。
        self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def _model_for(self, tier: ModelTier) -> str:
        return self.config.fast_model if tier == ModelTier.FAST else self.config.smart_model

    def complete(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.SMART,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """最基础的文本补全调用。返回模型输出的纯文本。

        失败时会按 `max_retries` 重试。重试是 Agent 鲁棒性的最低要求——
        面试官问"工具/模型调用挂了怎么办",这里就是答案的一部分。
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model_for(tier),
                    messages=messages,
                    temperature=(
                        temperature
                        if temperature is not None
                        else self.config.default_temperature
                    ),
                    max_tokens=max_tokens or self.config.max_tokens,
                )
                # 记录用量(有些兼容服务可能不返回 usage,做容错)。
                if resp.usage:
                    self.usage.add(
                        resp.usage.prompt_tokens or 0,
                        resp.usage.completion_tokens or 0,
                    )
                return (resp.choices[0].message.content or "").strip()
            except Exception as err:  # noqa: BLE001 —— 这里就是要兜住一切并重试
                last_err = err
                continue
        raise RuntimeError(f"LLM 调用在 {self.config.max_retries + 1} 次尝试后仍失败: {last_err}")

    def complete_json(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.SMART,
        system: str | None = None,
        temperature: float | None = None,
    ) -> Any:
        """要求模型返回 JSON,并做健壮解析。

        现实里模型经常"嘴上说只输出 JSON,实际还是套了 ```json 代码块或加了
        前言"。所以我们不能天真地直接 json.loads,而要先把 JSON 主体抠出来。
        这种"对模型不可靠输出的防御性解析"是 Agent 工程的日常。
        """
        sys = (system or "") + "\n\n你必须只输出合法的 JSON,不要包含任何解释文字或 Markdown 代码块标记。"
        raw = self.complete(
            prompt,
            tier=tier,
            system=sys.strip(),
            temperature=temperature if temperature is not None else 0.0,
        )
        return _extract_json(raw)


def _extract_json(text: str) -> Any:
    """从可能夹带杂质的文本里提取并解析 JSON。

    策略(由宽到严):
    1. 直接尝试 json.loads。
    2. 去掉 ```json ... ``` 代码块包裹后再试。
    3. 用正则抓取第一个 { ... } 或 [ ... ] 区块再试。
    全部失败则抛出带原文的异常,方便调试时定位是模型哪句话不规矩。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 去掉 markdown 代码块围栏
    fenced = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # 抓取第一个对象或数组
    match = re.search(r"(\{.*\}|\[.*\])", fenced, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从模型输出中解析出 JSON。原始输出:\n{text}")
