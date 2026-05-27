"""
环境配置加载 (Environment Config Loading)
==========================================

设计意图
--------
让项目能自动读取项目根目录下的 `.env` 文件,把里面的键值对加载进环境变量。
这样用户只需在 `.env` 里填好 OPENAI_API_KEY 等配置,直接运行即可,
不必每次在终端手动 export / $env: 设置。这是真实项目的标准做法。

为什么自己写一个极简加载器,而不强依赖 python-dotenv(工程鲁棒性)
--------------------------------------------------------------
优先使用 python-dotenv(功能完整、社区标准);但如果用户没装它,
就降级用一个内置的极简解析器兜底 —— 保证"忘了装 dotenv"也不会让项目跑不起来。
这种"优先用好工具、缺了也能优雅降级"的思路,和项目其它地方(搜索失败降级等)
是一致的。

幂等
----
load_env() 多次调用是安全的:已存在的环境变量不会被 .env 覆盖
(遵循 dotenv 的默认语义 —— 真实环境变量优先级高于 .env 文件)。
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_env() -> None:
    """加载项目根目录的 .env 到环境变量。幂等、可安全重复调用。"""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    env_path = _find_env_file()
    if env_path is None:
        return  # 没有 .env 也没关系,可能用户直接设了真实环境变量

    # 优先用 python-dotenv
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except ImportError:
        pass

    # 降级:极简解析器(够用,处理 KEY=VALUE、注释、引号)
    _load_env_fallback(env_path)


def _find_env_file() -> Path | None:
    """从当前工作目录向上找 .env(最多找 3 层),找到就返回路径。"""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents][:4]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load_env_fallback(env_path: Path) -> None:
    """无第三方库时的极简 .env 解析。真实环境变量优先,不覆盖已有。"""
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # 去掉可能的引号
            if key and key not in os.environ:  # 真实环境变量优先
                os.environ[key] = value
    except Exception:  # noqa: BLE001 —— 加载失败不应让程序崩溃
        pass