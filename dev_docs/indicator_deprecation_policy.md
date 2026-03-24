# Indicator Deprecation Policy

更新时间：2026-03-17（Asia/Shanghai）

## 元数据字段

- `version`: 当前指标版本（默认 `1.0.0`）。
- `status`: `active` / `deprecated` / `removed`。
- `deprecated_since`: 当 `status=deprecated` 时必填，格式建议 `YYYY-MM-DD`。
- `replacement`: 推荐替代指标名（可空）。
- `remove_after`: 当 `status=removed` 时必填，格式建议 `YYYY-MM-DD`。

## 弃用窗口规则

- 最短弃用窗口：30 天。
- 推荐弃用窗口：60-90 天（生产策略建议 90 天）。
- `deprecated` 阶段：
  - 仍可运行；
  - MCP catalog/detail 必须返回状态与替代项；
  - semantic/backtest 对旧 alias 打 warning（不静默）。
- `removed` 阶段：
  - MCP catalog 不应继续推荐；
  - 策略新建流程禁止自动选用；
  - 历史策略需给出替代建议。

## 发布流程建议

1. 标记 `status=deprecated`，填写 `deprecated_since/replacement`。
2. 发布后观察 warning 与使用量至少一个窗口。
3. 达到 `remove_after` 前至少一周通知。
4. 到期后切换 `status=removed`，并保留一版迁移文档。

