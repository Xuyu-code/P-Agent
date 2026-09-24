# 河湟皮影 Agent：公开文化资料 source cards

本目录只保存“自写摘要 + 来源元数据”，不保存外部网页全文、新闻配图、馆藏图像、视频或音频。

## 使用边界

- `heritage_public_facts/`：公开来源的事实摘要，适合用于内部 RAG 和带来源回答。
- 外部网页“可访问”不等于“可自由复制或再发布”。默认策略是改写事实、保留来源，不批量抓取全文。
- `license_status=needs_review` 的来源，在公开部署或商业化前必须逐条核对版权/转载许可。
- 项目自身的公开服务说明和接口说明单独放入 `project_facts`，不能伪装成传统文化史料。

## 建议的 RAG 元数据

每个 card 至少包含：

`source_type`, `title`, `publisher`, `date`, `source_domain`, `locator`, `license_status`, `license_note`, `confidence`, `claim_scope`。

## 回答策略

1. 事实卡有证据：回答并显示来源标题。
2. 多个来源只支持结构事实，不支持固定象征含义：只回答结构事实，不编造寓意。
3. 没有证据的具体问题：说明证据边界，并建议用户提供传承人、剧目、馆藏说明或授权文献。
