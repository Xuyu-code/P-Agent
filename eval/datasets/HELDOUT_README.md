# Held-out 泛化评测集合（冻结）

**冻结时间：2026-08-22，先于任何针对成本的 prompt/检索优化写定。**

`heldout_*.jsonl` 是冻结评测集合，用途只有一个：衡量系统对**没见过的表述**的
表现。因此纪律是——

1. 用例不与 `intent_routing.jsonl` / `qa_answerability.jsonl` /
   `slot_extraction.jsonl`（调优用例集）重复或仅做字面改写；
2. **一次跑分，不许回头修**：跑出的失败案例只能写进报告，不能用来改
   prompt、阈值或用例本身；要修就开新一轮并重新声明调优集/冻结集；
3. 应答题的 `must_cite` / `must_cite_any` 依据 `data_sources/` 卡片实际
   支持的事实手工标注，拒答题均为知识库无证据支撑或诱导编造的问题。

运行：`python -m eval.run_eval --heldout`（其余参数同主评测）。
