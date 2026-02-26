# Orchestrator 包协作说明（中文版）

本文档面向后续维护者，重点解释：

- 包内每个文件在一轮对话中的职责和调用关系
- 状态对象如何在文件之间传递
- 为什么当前 stress/deployment 还是“占位能力”
- 后续如何把 stress_test / deployment 变成真实可运行 phase（含改造清单与 TODO）

---

## 1. 包定位与入口

### 1.1 对外入口

- 对外仍使用：
  - `from apps.api.orchestration import ChatOrchestrator`
- 包出口在 `__init__.py`，导出：
  - `ChatOrchestrator`
  - `get_handler`（保持 monkeypatch 兼容）
  - `_OPENAI_STREAM_HARD_TIMEOUT_SECONDS`（测试可覆盖超时）

### 1.2 运行主链路

路由层只调用一个方法：

- `ChatOrchestrator.handle_message_stream(...)`

该方法内部会依次调用：

1. `core.py` 会话解析 + phase 边界处理
2. `prompt_builder.py` 组装 prompt/tools/context
3. `stream_handler.py` 消费 OpenAI 流并实时 SSE 转发
4. `postprocessor.py` 解析模型输出并调用 handler 后处理
5. `core.py` / `postprocessor.py` 触发 phase 迁移与持久化

---

## 2. 文件分工与协作关系

| 文件 | 核心职责 | 主要上游调用方 | 主要下游调用目标 |
|---|---|---|---|
| `core.py` | `ChatOrchestrator` 主流程、session 生命周期、phase transition 审计 | API routers | `prompt_builder` / `stream_handler` / `postprocessor` / `carryover` / `strategy_context` |
| `prompt_builder.py` | 构建 `PromptPieces`、runtime policy、tools、MCP context header | `core.handle_message_stream` | `handler.build_prompt`、`_hydrate_strategy_context` |
| `stream_handler.py` | OpenAI Responses stream 消费、事件透传、text delta 聚合、error 捕获 | `core.handle_message_stream` | `mcp_records`、SSE 事件发射 |
| `postprocessor.py` | 从全文提取 `AGENT_UI_JSON` / `AGENT_STATE_PATCH`，调用 handler `post_process`，处理 phase 切换后尾事件 | `core.handle_message_stream` | `genui_wrapper` / `fallback` / `carryover` |
| `genui_wrapper.py` | strategy/backtest 场景自动补齐 GenUI（`strategy_ref`、`backtest_charts`） | `postprocessor` | `engine.strategy`（draft backfill） |
| `mcp_records.py` | 将流式 MCP 事件合并为可落库工具调用记录，并去除“失败后重试成功”的噪音失败记录 | `stream_handler` | `postprocessor`（写入 assistant message.tool_calls） |
| `strategy_context.py` | strategy/stress 上下文补水（`strategy_id` 解析、归属校验）；当前还承担 legacy stress->strategy 重定向 | `core` / `prompt_builder` | DB 查询（`Strategy`）+ `core._transition_phase` |
| `carryover.py` | phase 切换时生成并消费 carryover memory；phase turn 计数 | `core` / `postprocessor` | DB 查询（历史 `Message`） |
| `fallback.py` | 空回复 fallback、停止准则占位提示（turn limit） | `postprocessor` | session metadata 更新 |
| `constants.py` | orchestrator 常量（tag/tool sets/limits） | 各 mixin | 无 |
| `types.py` | `_TurnPreparation` / `_TurnStreamState` / `_TurnPostProcessResult` 三个跨模块数据结构 | 各 mixin | 无 |
| `shared.py` | 汇总 import，统一 `from .shared import *` 的上下文 | 各 mixin | handler protocol / model / settings / util |

---

## 3. 一轮消息完整时序（细化版）

### 3.1 `core.handle_message_stream()` 前半段

1. `_resolve_session()`
   - 无 `session_id` 时新建 session（`create_session()`）
2. `_enforce_strategy_only_boundary()`
   - 当前若 session 停在 `stress_test`，会被系统重定向回 `strategy`
3. `_consume_phase_carryover_memory()`
   - 若 metadata 中存在本 phase carryover block，拼到用户消息前
4. 立即落库 user message，并 `_increment_phase_turn_count()`
5. SSE 发 `stream_start`

### 3.2 `prompt_builder._prepare_turn_context()`

1. 深拷贝并归一化 artifacts（确保按 phase 建块）
2. `_hydrate_strategy_context()`：
   - 尝试从 artifacts / metadata / 用户消息 UUID / DB 历史 strategy 解析 `strategy_id`
3. 解析 pre_strategy 的 `target_instrument` 旧值（用于后续 chart 自动注入判定）
4. `_resolve_runtime_policy()`：
   - phase 默认策略 + 用户请求 runtime policy 合并
5. `handler.build_prompt(ctx, user_message)` 拿到 `PromptPieces`
6. `_merge_tools()` 和 `_attach_mcp_context_headers()`
   - 对特定 `server_label` 注入 `x-mcp-context` JWT
7. 产出 `_TurnPreparation`

### 3.3 `stream_handler._stream_openai_and_collect()`

1. 组装 `stream_request_kwargs`
2. 打 trace（自动脱敏 header token）
3. `async for event in streamer.stream_events(...)`
4. 每个事件都会：
   - 透传 `openai_event` SSE
   - `mcp_records._collect_mcp_records_from_event()` 聚合 MCP 调用轨迹
   - 如是 `response.output_text.delta`，向前端发 `text_delta`
   - 如是 MCP 事件，再发 `mcp_event`
   - 如是 `response.completed`，刷新 `session.previous_response_id`
5. 超时/异常会写入 `stream_error_message` 与 `stream_error_detail`

### 3.4 `postprocessor._post_process_turn()`

1. `_extract_wrapped_payloads(full_text)`：
   - 提取 `<AGENT_UI_JSON>`、`<AGENT_STATE_PATCH>`
   - 清理 echo 的 `[SESSION STATE]` 和伪造 `mcp_xxx` 标签
2. 构建可持久化 MCP tool_calls（仅保留 success/failure，过滤可重试噪音失败）
3. GenUI 归一化 + 自动补齐：
   - `_maybe_auto_wrap_strategy_genui`
   - `_maybe_backfill_strategy_ref_from_validate_call`
   - `_maybe_auto_wrap_backtest_charts_genui`
4. 调用 `handler.post_process(...)`，得到 phase 语义结果
5. handler 过滤 GenUI + orchestrator 兜底：
   - `_ensure_required_choice_prompt_payload`
   - `_ensure_pre_strategy_chart_payload`
6. 若 `result.completed and result.next_phase`：
   - 调 `core._transition_phase()`
   - 记录 carryover memory（写 metadata）
7. 文本兜底链：
   - stream 中断 fallback
   - stop criteria 占位提示
   - 空回复 fallback
8. 产出 `_TurnPostProcessResult`

### 3.5 `postprocessor._emit_tail_events_and_persist()`

1. 落库 assistant message（含 `tool_calls`、`token_usage`、`response_id`）
2. 刷新会话标题
3. SSE 顺序发出：
   - `genui`（逐条）
   - `phase_change`（如发生）
   - 终态 `text_delta`（仅在流阶段未发完文本时）
   - `done`（包含 phase/status/missing_fields/usage/stream_error）

---

## 4. 三个跨模块状态对象（`types.py`）

### 4.1 `_TurnPreparation`

由 `prompt_builder` 生成，供 `stream_handler` 和 `postprocessor` 共用：

- 固定本轮 `phase_before`
- 固定 `handler`
- 固定 prompt/tools 组合
- 包含 artifacts 快照与 pre_strategy 旧 instrument 值

### 4.2 `_TurnStreamState`

由 `stream_handler` 逐事件填充，供 `postprocessor` 消费：

- `full_text`：增量拼接后的模型全文
- `completed_usage`：OpenAI usage
- `stream_error_*`：流中断诊断
- `mcp_call_records` + `mcp_call_order`：MCP 事件归并结果

### 4.3 `_TurnPostProcessResult`

由 `postprocessor` 产出，供 tail emit + persistence：

- 最终 assistant 文本
- 过滤后 GenUI
- 可持久化 tool_calls
- missing_fields / kyc_status / transitioned

---

## 5. 关键协作细节（易踩坑）

### 5.1 monkeypatch 兼容点（测试依赖）

- `core._resolve_handler_from_module()` 会从包级 `apps.api.orchestration.get_handler` 取 handler
- `stream_handler._resolve_stream_timeout_seconds()` 会读包级 `_OPENAI_STREAM_HARD_TIMEOUT_SECONDS`

这两个点保证了旧测试无需改导入路径也能 monkeypatch。

### 5.2 artifacts 的“相位键完整性”

- `_ensure_phase_keyed()` 会保证每个 workflow phase 都有默认 artifact block
- 若缺失，会用 `handler.init_artifacts()` 自动补齐

### 5.3 MCP 记录并非“原样存储”

- `mcp_records.py` 会跨事件聚合同一个 call
- 最终只落 success/failure
- 若同工具同参数“先失败后成功”，失败记录会被抑制

### 5.4 当前 stress_test 真实语义

虽然 `stress_test_handler` 存在，但 orchestrator 前置边界仍会把 session 拉回 `strategy`。  
所以 stress_test 目前是兼容态，不是主路径执行态。

---

## 6. 当前与 phase 相关的占位/TODO 清单（必须认领）

以下是后续实现真实 stress/deployment 前必须处理的占位逻辑：

- `apps/api/orchestration/strategy_context.py`
  - `_enforce_strategy_only_boundary()` 中有 `stress_test_disabled_redirect_to_strategy`
- `apps/api/agents/phases.py`
  - `VALID_TRANSITIONS` 目前不允许 `strategy -> stress_test`
- `apps/api/schemas/requests.py`
  - `StrategyConfirmRequest.advance_to_stress_test` 注释为 deprecated/ignored
- `apps/api/routes/strategies.py`
  - `_apply_strategy_confirm_metadata()` 明确忽略 `advance_to_stress_test`
- `apps/api/orchestration/fallback.py`
  - `stop_criteria_placeholder.performance_threshold_todo = True` 仍为占位
- `apps/api/agents/skills/stress_test/skills.md` 与 `stages/*.md`
  - 文案是 legacy placeholder，不是正式 stress 流程定义

---

## 7. 如何把 Stress Test Phase 做成“真实可用”

目标路径建议：

- `strategy -> stress_test -> deployment`
- 异常回退：`stress_test -> strategy`

### 7.1 必改现有文件

1. `apps/api/agents/phases.py`
- 放开 `Phase.STRATEGY -> Phase.STRESS_TEST`
- 增加 `Phase.STRESS_TEST -> Phase.DEPLOYMENT`

2. `apps/api/orchestration/strategy_context.py`
- 去掉或改造 `_enforce_strategy_only_boundary()` 的强制回退
- 仅对“非法 legacy 数据”做纠偏，不改 phase

3. `apps/api/agents/handlers/strategy_handler.py`
- 在合适条件（如 strategy 确认+回测完成）返回：
  - `completed=True`
  - `next_phase=Phase.STRESS_TEST.value`
- 并写清 `transition_reason`

4. `apps/api/routes/strategies.py` + `apps/api/schemas/requests.py`
- 让 `advance_to_stress_test` 从 ignored 变为真实触发条件之一
- 清理 `advance_to_stress_test_ignored` metadata 写入逻辑

5. `apps/api/orchestration/prompt_builder.py`
- 完善 `_build_stress_test_runtime_policy()` 的 stage 切换规则
- 把 stage 判定从“仅看 backtest_status”升级为完整状态机（例如 bootstrap -> scenario -> feedback）

6. `apps/api/agents/handlers/stress_test_handler.py`
- 增加 stress 专属字段（如 `scenario_set_id`, `stress_report_id`, `stress_test_decision`）
- 允许转入 deployment（而不是只回 strategy）

7. `apps/api/agents/skills/stress_test/*`
- 重写 skills 文案，去掉 legacy 描述
- stage markdown 从 2 段扩展为真实流程段

### 7.2 建议新增代码文件（最小可维护拆分）

建议在 orchestrator 包再拆两层相位专项能力，避免 `prompt_builder`/`postprocessor` 再膨胀：

- `apps/api/orchestration/stress_runtime.py`
  - stress phase 的 runtime policy、allowed tools、tool_choice 规则
- `apps/api/orchestration/stress_postprocess.py`
  - stress 工具结果自动包装（如 `stress_report` GenUI）、状态回填
- `apps/api/orchestration/deployment_runtime.py`
  - deployment phase tool 暴露策略与最小必需工具集
- `apps/api/orchestration/deployment_postprocess.py`
  - deployment 工具回执转 UI 与状态 patch 的统一逻辑

如采用以上拆分，应在 `core.ChatOrchestrator` 继承链新增对应 mixin。

### 7.3 需要同步新增/调整测试

- `tests/test_agents/test_orchestrator_runtime_policy.py`
  - 新增 strategy->stress、stress->deployment runtime policy 用例
- `tests/test_agents/test_phase_handlers.py`
  - 新增 stress 完成后进入 deployment 的断言
- 新增建议：`tests/test_api/test_stress_deployment_flow.py`
  - 覆盖 end-to-end phase_change 序列与 artifacts 演进
- `tests/test_api/test_strategy_confirm.py`
  - 将 `advance_to_stress_test_ignored` 相关断言改为真实转阶段断言

---

## 8. 如何把 Deployment Phase 做成“真实交付闭环”

当前 deployment handler 只处理 `deployment_status` 字段。要变成真实交付，需要补“执行层 + 可观测”。

### 8.1 必改现有文件

1. `apps/api/agents/handlers/deployment_handler.py`
- 扩展字段：如 `deployment_target`, `risk_guard_status`, `rollout_status`, `deployment_id`
- `post_process` 支持更多分支：
  - `ready -> deployed -> completed`
  - `blocked -> strategy`（带 blocker code）

2. `apps/api/orchestration/prompt_builder.py`
- 为 deployment 加 runtime policy（工具和 stage）

3. `apps/api/orchestration/fallback.py`
- 增加 deployment 缺字段场景 fallback 文本

4. `apps/api/agents/skills/deployment/skills.md`
- 增加真实部署 SOP、失败回滚语义、输出契约

### 8.2 建议新增跨层文件（非 orchestrator 包内）

- `packages/infra/providers/trading/adapters/alpaca_trading.py`
- `packages/infra/providers/trading/adapters/alpaca_trading.py`
- `packages/domain/trading/runtime/runtime_service.py`
- `apps/mcp/domains/trading/tools.py`

orchestrator 只做编排，不直接持有交易执行细节；通过 MCP 或 service 层能力调用。

### 8.3 观测与审计建议

- 在 `tool_calls` 中保留部署请求 id / 回执 id
- `PhaseTransition.metadata_` 写明部署触发原因和 blocker
- done 事件扩展 deployment 状态摘要（避免前端再次拉取才知道结果）

---

## 9. 推荐执行顺序（落地 stress + deployment）

1. 先改 `phases.py` 转换图与 `strategy_context.py` 边界重定向
2. 再改 `strategy_handler` 和 `stress_test_handler` 的 phase 输出语义
3. 再完善 orchestrator runtime/postprocess 专项模块
4. 最后接 deployment 执行层与 MCP 工具
5. 每一步都跑回归（见下节）

---

## 10. 回归测试基线

当前必须通过的基线：

- `uv run pytest -q tests/test_agents/test_orchestrator_runtime_policy.py`
- `uv run pytest -q tests/test_api/test_stream_empty_output_fallback.py tests/test_api/test_stream_error_detail.py tests/test_api/test_mcp_history_persistence.py`
- `uv run pytest -q tests/test_api/test_kyc_flow.py tests/test_api/test_pre_strategy_robustness.py tests/test_api/test_strategy_confirm.py`
- `uv run pytest -q tests/test_agents/test_phase_handlers.py`

当 stress/deployment 启用后，建议新增一组“phase 主链路”端到端测试并纳入必跑清单。
