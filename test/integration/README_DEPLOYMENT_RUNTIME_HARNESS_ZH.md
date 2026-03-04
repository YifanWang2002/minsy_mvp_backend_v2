# Deployment Runtime Harness 中文说明

## 1. 这套 Harness 是做什么的

这套 `Deployment Runtime Harness` 是一组面向 deployment 阶段及之后的全链路集成测试与诊断工具，目标不是只验证单个接口是否返回 `200`，而是把下面这些链路串起来做真实验证：

- 策略已入库后，创建 deployment
- deployment `start / pause / stop`
- deployment 运行中的 `portfolio / orders / fills`
- 手动交易动作 `open / close / reduce / stop`
- AI chat 入口
- MCP trading tools 入口
- trade approval 审批链路
- Telegram webhook / social connector 绑定链路
- notification outbox / worker 分发链路
- 数据库、Redis、Celery 的状态观测

它的定位是：

- 回归测试：确认 deployment 阶段的真实行为没有被改坏
- 诊断工具：当你遇到“按钮点了没请求”“请求太慢”“状态没推进”时，给出可复现的证据链
- 性能基线：把每一步的耗时、DB 状态、队列状态、Redis runtime state 一起记录下来


## 2. 相关代码位置

核心代码分成三层：

- Harness 实现：
  - `backend/test/_support/deployment_runtime_harness.py`
- Harness live 测试入口：
  - `backend/test/integration/test_deployment_runtime_harness_live.py`
- 前端真实点击链验证：
  - `frontend/integration_test/portfolio_deployment_harness_real_test.dart`

本轮为了支持 harness 做的前端请求追踪和状态展示，主要在：

- `frontend/lib/services/dio_client.dart`
- `frontend/lib/services/portfolio_api_service.dart`
- `frontend/lib/providers/portfolio_provider.dart`
- `frontend/lib/screens/portfolio_screen.dart`
- `frontend/lib/screens/portfolio/portfolio_components.dart`
- `frontend/lib/screens/portfolio/portfolio_helpers.dart`


## 3. 它目前覆盖哪些能力

### v1：REST 控制面全链路

场景方法：

- `DeploymentRuntimeHarness.run_v1_rest_scenario()`

覆盖内容：

- 注册测试用户
- 创建聊天 thread
- 创建可 deploy 的策略
- 创建内置 sandbox broker account
- 创建 deployment
- `start deployment`
- 拉取 `portfolio`
- 拉取 `orders`
- 拉取 `fills`
- 提交一次 UI 等价的 `manual open long`
- `pause deployment`
- `stop deployment`
- 采集 DB / Redis / Celery 状态
- 写出 JSON/Markdown 报告

这个场景主要用来验证：

- deployment 生命周期是否正常
- 手动单是否真正落到了后端
- `GET /portfolio` 的耗时基线
- 手动单后数据库里是否真的生成了 `manual_trade_actions`


### v1.1：Chat + MCP 扩展链路

场景方法：

- `DeploymentRuntimeHarness.run_v1_1_chat_mcp_scenario()`

覆盖内容：

- 先走一次真实 `chat/send-openai-stream`
- 再走 MCP trading tools：
  - `trading_list_deployments`
  - `trading_start_deployment`
  - `trading_get_orders`
  - `trading_pause_deployment`
  - `trading_stop_deployment`
- 最后继续采集 DB / Redis / Celery 状态并输出报告

这个场景主要用来验证：

- AI 入口是否能正常响应
- MCP server 是否连通
- MCP 工具触发 deployment 控制时，链路是否完整


### v1.2：审批 + 通知 + IM 链路

场景方法：

- `DeploymentRuntimeHarness.run_v1_2_approval_notification_scenario()`

覆盖内容：

- 打开通知偏好
- 把交易偏好切到 `approval_required`
- 启动 deployment
- 建立 Telegram binding
  - 优先尝试真实 connect-link + webhook `/start`
  - 如果当前环境不适合，再走 seeded fallback
- 直接 seed 一个待审批请求
- 列出审批列表
- 触发 Telegram callback 审批
  - 如果 callback 没把状态推进，再自动走 API fallback approve
- 执行通知分发
- inline 执行 `approved open`
- 拉取 social connector 和活动列表
- 最后采集 DB / Redis / Celery 状态并输出报告

这个场景主要用来验证：

- approval request 是否被创建
- webhook 是不是“真的处理成功”，而不是只返回 HTTP 200
- notification outbox 是否真的进入分发
- 审批后执行是否成功推进到了订单层


## 4. Harness 的内部结构

`deployment_runtime_harness.py` 不是一坨脚本，它内部按 Driver / Observer / Reporter 分层：

### Driver

- `ApiDriver`
  - 通过 FastAPI `TestClient` 调真实 REST 路由
- `ChatDriver`
  - 调用 `/api/v1/chat/send-openai-stream` 并解析 SSE
- `McpDriver`
  - 直接对 `http://127.0.0.1:8110/trading/mcp` 发 MCP tool call
- `DomainDriver`
  - 做测试专用 seed，例如 trade approval / telegram binding / worker 直跑
- `WebhookDriver`
  - 直接构造并投递 Telegram webhook payload

### Observer

- `DbObserver`
  - 采集关键表计数和最新记录
- `RedisObserver`
  - 采集 deployment runtime state 和关键队列长度
- `CeleryObserver`
  - 采集 worker `ping` 和 `stats`

### Reporter

- `HarnessReporter`
  - 把本次执行写成 JSON 报告
  - 同时写一份方便人工阅读的 Markdown 报告


## 5. 它会观测哪些系统状态

### 数据库

`DbObserver` 当前会统计并记录这些表：

- `deployments`
- `deployment_runs`
- `manual_trade_actions`
- `orders`
- `fills`
- `positions`
- `pnl_snapshots`
- `signal_events`
- `trade_approval_requests`
- `trading_event_outbox`
- `notification_outbox`
- `notification_delivery_attempts`
- `social_connector_bindings`
- `social_connector_link_intents`
- `social_connector_activities`

同时还会记录最近一条：

- `latest_manual_action`
- `latest_trade_approval`
- `latest_trading_event`
- `latest_notification`


### Redis

当前会观测：

- `paper_trading:runtime_state:{deployment_id}`
- `paper_trading:runtime_state:__live_trading_health__`
- 队列长度：
  - `paper_trading`
  - `trade_approval`
  - `notifications`
  - `market_data`
  - `maintenance`


### Celery

当前会观测：

- `inspect.ping()`
- `inspect.stats()`

这可以帮助判断：

- worker 是否在线
- 队列层是不是已经彻底挂了


## 6. 报告产物在哪里

每次成功执行后，harness 都会自动写两份报告到：

- `backend/runtime/harness_reports/`

命名规则示例：

- `deployment_runtime_v1_YYYYMMDDTHHMMSSZ.json`
- `deployment_runtime_v1_YYYYMMDDTHHMMSSZ.md`
- `deployment_runtime_v1_1_YYYYMMDDTHHMMSSZ.json`
- `deployment_runtime_v1_1_YYYYMMDDTHHMMSSZ.md`
- `deployment_runtime_v1_2_YYYYMMDDTHHMMSSZ.json`
- `deployment_runtime_v1_2_YYYYMMDDTHHMMSSZ.md`

JSON 适合做机器处理、二次分析、自动告警。  
Markdown 适合人工排障和快速查看。

Markdown 报告通常包含：

- 场景名
- 开始/结束时间
- deployment / user 上下文
- 每一步的：
  - 名称
  - driver
  - method
  - target
  - 状态码
  - 耗时
- DB 前后快照
- Redis probe
- Celery probe
- 附加 artifacts


## 7. 如何运行

### 运行全部 Harness Live 场景

在仓库根目录下执行：

```bash
cd backend
./.venv/bin/python -m pytest -q test/integration/test_deployment_runtime_harness_live.py
```

这会按顺序跑：

- `test_000_deployment_runtime_harness_v1_live`
- `test_010_deployment_runtime_harness_v1_1_live`
- `test_020_deployment_runtime_harness_v1_2_live`


### 只跑某一个场景

```bash
cd backend
./.venv/bin/python -m pytest -q test/integration/test_deployment_runtime_harness_live.py -k test_000_deployment_runtime_harness_v1_live
```

或者：

```bash
cd backend
./.venv/bin/python -m pytest -q test/integration/test_deployment_runtime_harness_live.py -k test_010_deployment_runtime_harness_v1_1_live
```

或者：

```bash
cd backend
./.venv/bin/python -m pytest -q test/integration/test_deployment_runtime_harness_live.py -k test_020_deployment_runtime_harness_v1_2_live
```


### 运行前端真实点击链验证

```bash
cd frontend
flutter test -d macos integration_test/portfolio_deployment_harness_real_test.dart
```

这个测试会真实验证：

- deploy strategy
- start
- open long
- stop

并通过 `Dio` trace 证明请求确实发出了。


### 运行前端回归测试

```bash
cd frontend
flutter test test/portfolio_provider_test.dart test/portfolio_screen_test.dart
flutter test -d chrome test/portfolio_provider_test.dart test/portfolio_screen_test.dart
dart analyze lib test
```


## 8. 运行前的环境要求

运行这套 harness 前，默认需要这些依赖是可用的：

- backend API 可用
- PostgreSQL 可用
- Redis 可用
- Celery worker 至少有基本可用性
- MCP server 在 `127.0.0.1:8110` 可达（仅 v1.1 需要）

默认假设：

- backend 测试环境已经可以通过 `TestClient` 建立应用上下文
- `backend/.venv` 已安装测试依赖
- 本地配置里允许注册测试用户、创建 sandbox broker account、创建 deployment

如果环境不满足，最常见的现象是：

- `/health` 不稳定
- `MCP tool call failed`
- Redis/Celery probe 返回空或错误
- Telegram callback 返回 200 但 `processed = false`


## 9. 如何解读常见问题

### 情况一：前端点了按钮，但看起来没反应

先看：

- 前端 `Dio` trace 是否有 `POST /deployments/{id}/manual-actions`
- Harness 报告里 `manual_open_long` 这一步是否存在
- `db_after.latest_manual_action.status` 是什么

常见解读：

- `executing` + `waiting_for_runtime_lock`
  - 说明不是没发请求，而是 runtime 正忙，已进入自动重试
- `rejected` + `deployment_locked_timeout`
  - 说明锁冲突重试超时
- `rejected` + `order_rejected`
  - 说明已进入执行，但订单被 broker/风控拒绝


### 情况二：`/portfolio` 太慢

先看：

- Harness 报告里 `get_portfolio.duration_ms`
- 同一份报告里的 Redis runtime state
- DB 里 `pnl_snapshots` 是否已有快照

当前代码已经做了缓存快读：

- 非 active deployment 优先走缓存
- active deployment 在最近快照足够新时优先走缓存

如果仍然慢，说明问题更可能在：

- broker provider
- position 数量过多
- 背景刷新逻辑
- 前端批量刷新策略


### 情况三：Telegram webhook 返回成功，但审批状态没变化

先看：

- webhook 步骤的返回体里 `processed`
- `artifacts.approval_after_callback`
- `db_after.latest_trade_approval`

现在 webhook 路由不会再假装总是成功：

- `{"ok": true, "processed": true}` 表示已处理
- `{"ok": true, "processed": false}` 表示请求格式合法，但没有实际推进状态
- `{"ok": false, "processed": false, "error": "webhook_processing_failed"}` 表示处理过程抛异常


## 10. 这套 Harness 不做什么

当前这套 harness 的目标是 deployment 阶段及其后的关键链路，不覆盖所有研发面：

- 它不是压力测试框架
- 它不是 UI 视觉回归测试
- 它不是完整的浏览器自动化（当前前端主验证仍以 Flutter integration test 为主）
- 它不替代单元测试或纯 API 契约测试

它更适合回答的是：

- 这条真实链路通不通
- 卡在前端、后端、队列、DB、Redis、通知链的哪一层
- 当前版本是否引入了 deployment 阶段的回归


## 11. 如何扩展新场景

如果你要继续扩展这套 harness，建议遵循现在的结构：

1. 先决定是新建一个 `run_vX_xxx_scenario()`，还是在现有场景里补步骤。
2. 所有“动作”优先放到 Driver 中：
   - REST 走 `ApiDriver`
   - Chat 走 `ChatDriver`
   - MCP 走 `McpDriver`
   - 测试 seed/worker 直跑走 `DomainDriver`
   - Webhook 走 `WebhookDriver`
3. 所有“观测”优先放到 Observer 中：
   - DB 用 `DbObserver`
   - Redis 用 `RedisObserver`
   - Celery 用 `CeleryObserver`
4. 新场景必须最终产出 `HarnessReport`，保证报告结构不碎片化。
5. 如果新增了真实业务链路，最好同步补一个 `pytest` live 用例。


## 12. 当前推荐的使用方式

日常开发里建议这样用：

1. 改 deployment 或 portfolio 相关后端逻辑后，先跑：
   - `test_deployment_runtime_harness_live.py`
2. 改前端 portfolio 交互后，再跑：
   - `portfolio_screen_test.dart`
   - `portfolio_provider_test.dart`
   - `portfolio_deployment_harness_real_test.dart`
3. 遇到“看起来没请求 / 没反应 / 太慢”的问题时：
   - 先看最新 `backend/runtime/harness_reports/*.md`
   - 再对照前端 `Dio` trace 和浏览器控制台日志

这样可以最快定位到底是：

- 前端没有发请求
- 请求发了，但被后端业务拒绝
- 请求进入了队列，但卡在 worker
- 状态已经推进，但 UI 没有正确反馈
