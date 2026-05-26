"""
网页抓取与定向摘要工具 (Fetch & Summarize Tool)
================================================

设计意图
--------
搜索只给我们一堆 URL 和短 snippet。要真正获取信息,得抓取网页正文。但网页正文动辄
几千上万字,如果直接塞进上下文,既烧 token 又稀释信息。所以这一步做两件事:

1. **抓取并清洗正文**:去掉导航、脚本、广告等噪音,只留可读文本。
2. **定向摘要 (query-focused summarization)**:用一个**便宜的小模型 (FAST tier)**,
   针对"当前这个子问题"对正文做摘要,只抽取与子问题相关的内容。

为什么这样设计(核心面试谈资)
----------------------------
这是整个项目里最能体现"成本意识"的地方,务必讲清楚:

- **不喂全文,只喂摘要**:报告生成阶段如果直接读 15 篇网页全文,token 会爆炸。
  我们用小模型先把每篇压缩成几百字的相关摘要,大模型最后只综合这些摘要。
- **用 FAST 档而非 SMART 档做摘要**:摘要是"读 + 抽取"的简单任务,小模型足矣。
  把高频任务交给便宜模型,是模型分级策略的直接落地。
- **定向(query-focused)而非泛泛摘要**:同一篇网页,对不同子问题应抽取不同内容。
  让摘要"带着问题去读",信息密度更高,也减少后续幻觉(因为材料更聚焦)。

可以算一笔账给面试官:假设一次研究抓 15 个网页、每篇正文 3000 token。
- 朴素方案:把 15×3000=45000 token 全喂给贵模型 → 巨贵。
- 本方案:每篇用便宜模型压到 ~300 token,贵模型只读 15×300=4500 token → 成本大幅下降。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.client import LLMClient, ModelTier


@dataclass
class Finding:
    """一条"发现":从某个来源针对某子问题提取出的相关信息。

    `source_url` 是引用溯源的关键 —— 它会一路带到报告生成阶段,
    保证报告里的每个引用都指向真实抓取过的网页,而不是模型编造的链接。
    """

    content: str
    source_url: str
    source_title: str


# 抓取正文时的超时(秒)。网页抓取是慢操作,必须设超时防止卡死整个流程。
_FETCH_TIMEOUT = 10
# 正文截断上限(字符)。超长网页只取前面一段喂给摘要模型,进一步控制成本。
_MAX_CONTENT_CHARS = 6000


class FetchSummarizeTool:
    """抓取网页并做定向摘要。

    用法::

        tool = FetchSummarizeTool(llm_client)
        finding = tool.run(url="https://...", title="...", sub_question="...")
        # finding 为 None 表示抓取失败或内容不相关
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _fetch_text(self, url: str) -> str | None:
        """抓取网页并提取纯文本正文。失败返回 None。"""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as err:
            raise RuntimeError(
                "未安装抓取依赖。请 `pip install requests beautifulsoup4`。"
            ) from err

        try:
            resp = requests.get(
                url,
                timeout=_FETCH_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0 (DeepResearchAgent)"},
            )
            resp.raise_for_status()
        except Exception as err:  # noqa: BLE001
            print(f"[FetchSummarize] 抓取失败,跳过该来源。url={url} err={err}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        # 去掉明显的噪音标签
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # 压缩空白行
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned = "\n".join(lines)
        return cleaned[:_MAX_CONTENT_CHARS] if cleaned else None

    def run(self, *, url: str, title: str, sub_question: str) -> Finding | None:
        """抓取 url 并针对 sub_question 做定向摘要。

        返回 Finding;若抓取失败或模型判断内容无关,则返回 None
        (上层会自然地忽略它,实现优雅降级)。
        """
        content = self._fetch_text(url)
        if not content:
            return None

        # 定向摘要:用 FAST 档小模型。提示词里明确两件事:
        # 1) 只抽取与子问题相关的内容;2) 如果整篇都不相关,明确说 IRRELEVANT。
        prompt = (
            f"下面是一篇网页的正文。请只抽取其中与这个子问题相关的关键信息,"
            f"用简洁的要点中文概括(150 字以内)。\n\n"
            f"子问题:{sub_question}\n\n"
            f"网页正文:\n{content}\n\n"
            f"如果整篇网页与子问题完全无关,只回复一个词:IRRELEVANT。"
        )
        summary = self.llm.complete(prompt, tier=ModelTier.FAST, temperature=0.0)

        if not summary or "IRRELEVANT" in summary.upper():
            return None

        return Finding(content=summary, source_url=url, source_title=title)
