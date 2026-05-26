# 示例运行输出（Sample Run）

下面是 `python -m deep_research "对比主流开源向量数据库的优劣"` 的**示意输出**，
用于让仓库访客直观了解 Agent 的运行轨迹和产出形态。

> 注：这是格式示意。真实运行时引用来源为实际抓取到的 URL，内容随搜索结果变化。

## 运行轨迹（Trace）

```
🎯 研究问题：对比主流开源向量数据库的优劣
📋 规划出 3 个子问题：
   - Milvus 的架构特点、性能与适用场景
   - Qdrant 与 Weaviate 的核心差异
   - 主流开源向量数据库的索引算法与可扩展性对比

🔬 开始研究子问题：Milvus 的架构特点、性能与适用场景
   [第 1 轮] 搜索查询：Milvus vector database architecture performance
      ✓ 收录：Milvus 官方文档 - 架构概览
      ✓ 收录：向量数据库性能基准对比
   ✅ 该子问题信息已充分（共 3 条）。

🔬 开始研究子问题：Qdrant 与 Weaviate 的核心差异
   [第 1 轮] 搜索查询：Qdrant vs Weaviate comparison
      ✓ 收录：Qdrant 与 Weaviate 对比文章
   [第 2 轮] 搜索查询：Weaviate hybrid search filtering capabilities
      ✓ 收录：Weaviate 混合检索能力介绍
   ✅ 该子问题信息已充分（共 3 条）。

🔬 开始研究子问题：主流开源向量数据库的索引算法与可扩展性对比
   [第 1 轮] 搜索查询：vector database HNSW IVF index scalability
      ✓ 收录：HNSW 与 IVF 索引原理对比
   ⏹️ 达到搜索次数上限 (2)，停止该子问题（共 2 条发现）。

✅ 全局反思：覆盖充分，无需补充。
📝 正在综合所有发现生成报告...
💰 本次研究共调用模型 14 次，消耗 token：18243（prompt 15102 + completion 3141）
```

## 产出报告（节选示意）

```
## 概述
本报告对比三款主流开源向量数据库 Milvus、Qdrant、Weaviate 在架构、
检索能力与可扩展性上的差异。

## 架构与定位
Milvus 采用存算分离的分布式架构，适合大规模生产部署 [1][2]。
Qdrant 以 Rust 实现，单机性能与资源效率突出 [3]。
Weaviate 内置混合检索与模块化向量化能力 [4]。

## 索引与检索
三者均支持 HNSW 索引；在过滤检索场景下 Weaviate 的混合检索更成熟 [4][5]。
...

## 选型建议
- 超大规模、需水平扩展：优先 Milvus [1][2]。
- 资源敏感、单机高性能：优先 Qdrant [3]。
- 需要开箱即用的混合检索：优先 Weaviate [4]。

---

## 参考来源

[1] Milvus 官方文档 - 架构概览 - https://...
[2] 向量数据库性能基准对比 - https://...
[3] Qdrant 与 Weaviate 对比文章 - https://...
[4] Weaviate 混合检索能力介绍 - https://...
[5] HNSW 与 IVF 索引原理对比 - https://...
```

注意报告里每个论断后的 `[n]` 都对应末尾真实抓取到的来源 URL——这就是
**可溯源引用**机制的产物。
