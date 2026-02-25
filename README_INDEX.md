---
index_name: backend-repo-index
generated_at_utc: 2026-02-21T18:39:20Z
repo_root: /Users/yifanwang/minsy_mvp_remastered/backend
scan_scope:
  - src/**/*.py
  - scripts/**/*.py
  - tests/**/*.py
symbol_index_file: docs/backend_symbol_index.json
symbol_index_stats:
  files_total: 277
  src_files: 174
  src_symbols: 986
  tests_files: 98
  scripts_files: 5
search_strategy:
  primary: symbol_index
  secondary: ripgrep_full_text
  vector_index: disabled
---

# Backend 代码库索引 README

> 本文档面向“人读 + 机器读”：
> - 人读：按模块给出职责、关键符号、运行/部署入口、风险点。
> - 机器读：顶部 YAML 元数据 + `docs/backend_symbol_index.json` 全量符号索引。

## TOC
- [1. 项目结构与模块分布](#1-项目结构与模块分布)
- [2. 模块关键函数/类索引（带路径与行号）](#2-模块关键函数类索引带路径与行号)
- [3. 代码风格与工程约定摘要](#3-代码风格与工程约定摘要)
- [4. 运行/部署/本地开发要点](#4-运行部署本地开发要点)
- [5. 未实现/技术债/风险点（含证据）](#5-未实现技术债风险点含证据)
- [6. 推荐索引/检索策略（简短）](#6-推荐索引检索策略简短)
- [7. 输出格式与导航约定](#7-输出格式与导航约定)

## 1. 项目结构与模块分布

### 1.1 目录/包布局（backend）

```text
backend/
├── src/
│   ├── main.py, config.py, dependencies.py
│   ├── agents/          # 会话阶段编排 + phase handlers + skills
│   ├── api/             # FastAPI routers/middleware/schemas
│   ├── engine/          # strategy/backtest/execution/market_data/feature/pnl
│   ├── mcp/             # MCP server + domain tools (strategy/backtest/market/...)
│   ├── models/          # SQLAlchemy ORM + db/redis bootstrap
│   ├── services/        # auth/openai stream/session title/social/telegram
│   ├── workers/         # Celery app + tasks
│   ├── observability/   # sentry + openai cost tracking
│   └── util/            # logging / data setup / trace util
├── tests/               # pytest 测试套件
├── scripts/             # 运维/回归脚本
├── deploy/systemd/      # systemd unit（flower）
├── .github/workflows/   # CI/CD（deploy）
├── data/                # parquet 数据
└── docs/backend_symbol_index.json  # 机器可读符号索引
```

### 1.2 模块密度（按 `src/` 一级包）

| 模块 | Python 文件数 | 顶层符号数 | 说明 |
|---|---:|---:|---|
| `src/engine` | 65 | 408 | 回测/执行/行情/策略核心引擎 |
| `src/api` | 22 | 177 | HTTP 接口、SSE、鉴权、schema |
| `src/mcp` | 15 | 132 | MCP 域路由与工具注册 |
| `src/agents` | 29 | 103 | 对话编排、phase handler、技能模板 |
| `src/services` | 6 | 49 | 业务服务层（OpenAI/Auth/Telegram） |
| `src/models` | 20 | 37 | ORM 模型与基础设施连接 |
| `src/workers` | 6 | 25 | Celery 队列任务 |
| `src/observability` | 3 | 22 | Sentry 与成本追踪 |
| `src/util` | 4 | 26 | 日志、数据准备、trace |

### 1.3 服务分布（进程/端口/网关）

| 服务 | 入口 | 端口/路由 | 证据 |
|---|---|---|---|
| FastAPI | `src.main:app` | `:8000` + `/api/v1/*` | `src/main.py:65`, `src/main.py:96` |
| MCP Strategy | `python -m src.mcp.server --domain strategy` | `:8111` | `.github/workflows/deploy.yml:102`, `Caddyfile:21` |
| MCP Backtest | 同上 `--domain backtest` | `:8112` | `.github/workflows/deploy.yml:102`, `Caddyfile:29` |
| MCP Market | 同上 `--domain market` | `:8113` | `.github/workflows/deploy.yml:102`, `Caddyfile:37` |
| MCP Stress/Trading | `--domain stress/trading` | `:8114/:8115` | `Caddyfile:45`, `Caddyfile:53` |
| Celery Worker/Beat | `src.workers.celery_app.celery_app` | queue: `backtest/market_data/paper_trading/maintenance` | `src/workers/celery_app.py:36`, `src/workers/celery_app.py:74` |
| Flower | systemd unit | `:5555` | `deploy/systemd/minsy-flower.service:10`, `Caddyfile:74` |

## 2. 模块关键函数/类索引（带路径与行号）

### 2.1 App 启动与配置模块

| 模块 | 关键函数/类 | 说明 |
|---|---|---|
| `src/main.py` | `create_app` (`src/main.py:65`) | FastAPI app factory，挂载路由/中间件/生命周期 |
| `src/main.py` | `lifespan` (`src/main.py:23`) | 启动时初始化数据、Postgres、Redis；关闭时释放连接 |
| `src/config.py` | `Settings` (`src/config.py:17`) | 全局运行配置（Pydantic Settings） |
| `src/config.py` | `get_settings` (`src/config.py:772`) | 配置单例缓存入口 |
| `src/dependencies.py` | `get_db` (`src/dependencies.py:18`) | API DB session DI |
| `src/dependencies.py` | `get_redis` (`src/dependencies.py:23`) | Redis client DI |

### 2.2 Agents 编排模块

| 模块 | 关键函数/类 | 说明 |
|---|---|---|
| `src/agents/orchestrator/core.py` | `ChatOrchestrator` (`src/agents/orchestrator/core.py:289`) | 核心 orchestrator 组合类 |
| `src/agents/orchestrator/core.py` | `OrchestratorCoreMixin` (`src/agents/orchestrator/core.py:16`) | 会话创建、流式处理主链路 |
| `src/agents/orchestrator/stream_handler.py` | `StreamHandlerMixin` (`src/agents/orchestrator/stream_handler.py:8`) | OpenAI 事件流转 SSE |
| `src/agents/orchestrator/prompt_builder.py` | `PromptBuilderMixin` (`src/agents/orchestrator/prompt_builder.py:8`) | 提示词与工具选择拼装 |
| `src/agents/orchestrator/postprocessor.py` | `PostProcessorMixin` (`src/agents/orchestrator/postprocessor.py:8`) | 输出后处理、artifact 同步 |
| `src/agents/orchestrator/strategy_context.py` | `StrategyContextMixin` (`src/agents/orchestrator/strategy_context.py:8`) | strategy/stress 上下文注入 |
| `src/agents/handlers/kyc_handler.py` | `KYCHandler` (`src/agents/handlers/kyc_handler.py:63`) | KYC phase 校验与引导 |
| `src/agents/handlers/pre_strategy_handler.py` | `PreStrategyHandler` (`src/agents/handlers/pre_strategy_handler.py:93`) | 策略前置需求收集 |
| `src/agents/handlers/strategy_handler.py` | `StrategyHandler` (`src/agents/handlers/strategy_handler.py:24`) | strategy phase 解析与推进 |
| `src/agents/handlers/stub_handler.py` | `StubHandler` (`src/agents/handlers/stub_handler.py:22`) | 未实现 phase 占位处理 |

### 2.3 API 模块

| 模块 | 关键函数/类 | 说明 |
|---|---|---|
| `src/api/middleware/auth.py` | `get_current_user` (`src/api/middleware/auth.py:15`) | JWT 鉴权依赖 |
| `src/api/middleware/rate_limit.py` | `RateLimiter` (`src/api/middleware/rate_limit.py:16`) | 端点级限流 |
| `src/api/middleware/sentry_http_status.py` | `SentryHTTPStatusMiddleware` (`src/api/middleware/sentry_http_status.py:112`) | HTTP 状态码到 Sentry 事件 |
| `src/api/routers/auth.py` | `register` (`src/api/routers/auth.py:68`), `login` (`src/api/routers/auth.py:88`) | 用户认证流程 |
| `src/api/routers/chat.py` | `new_thread` (`src/api/routers/chat.py:38`), `send_message_stream` (`src/api/routers/chat.py:63`) | 会话创建与 SSE 聊天 |
| `src/api/routers/strategies.py` | `confirm_strategy` (`src/api/routers/strategies.py:286`) | DSL 确认/入库/自动回测 |
| `src/api/routers/backtests.py` | `get_backtest_analysis` (`src/api/routers/backtests.py:58`) | 回测分析结果切片 |
| `src/api/routers/deployments.py` | `create_deployment` (`src/api/routers/deployments.py:211`), `start_deployment` (`src/api/routers/deployments.py:368`) | 运行部署生命周期 |
| `src/api/routers/market_data.py` | `get_latest_quote` (`src/api/routers/market_data.py:36`) | 行情查询 |
| `src/api/routers/portfolio.py` | `get_deployment_portfolio` (`src/api/routers/portfolio.py:57`) | 组合快照 |
| `src/api/routers/sessions.py` | `list_sessions` (`src/api/routers/sessions.py:24`) | 会话列表/归档 |
| `src/api/routers/trading_stream.py` | `stream_deployment` (`src/api/routers/trading_stream.py:121`) | 交易部署状态流 |
| `src/api/routers/social_connectors.py` | `telegram_webhook` (`src/api/routers/social_connectors.py:121`) | 社交连接器 webhook |

### 2.4 Engine 模块

#### 2.4.1 Strategy 子模块

| 模块 | 关键函数/类 | 说明 |
|---|---|---|
| `src/engine/strategy/pipeline.py` | `validate_strategy_payload` (`src/engine/strategy/pipeline.py:27`) | DSL schema + semantic 校验 |
| `src/engine/strategy/pipeline.py` | `parse_strategy_payload` (`src/engine/strategy/pipeline.py:47`) | DSL -> 可执行结构 |
| `src/engine/strategy/parser.py` | `build_parsed_strategy` (`src/engine/strategy/parser.py:15`) | 构建解析对象 |
| `src/engine/strategy/storage.py` | `upsert_strategy_dsl` (`src/engine/strategy/storage.py:369`) | 版本化持久化 |
| `src/engine/strategy/storage.py` | `get_strategy_or_raise` (`src/engine/strategy/storage.py:251`) | 查询与异常封装 |
| `src/engine/strategy/draft_store.py` | `create_strategy_draft` (`src/engine/strategy/draft_store.py:113`) | Redis 草稿缓存 |

#### 2.4.2 Backtest 子模块

| 模块 | 关键函数/类 | 说明 |
|---|---|---|
| `src/engine/backtest/engine.py` | `EventDrivenBacktestEngine` (`src/engine/backtest/engine.py:68`) | 事件驱动回测主引擎 |
| `src/engine/backtest/service.py` | `create_backtest_job` (`src/engine/backtest/service.py:72`) | 回测任务入队前持久化 |
| `src/engine/backtest/service.py` | `execute_backtest_job` (`src/engine/backtest/service.py:167`) | 执行与结果写回 |
| `src/engine/backtest/analytics.py` | `build_backtest_overview` (`src/engine/backtest/analytics.py:42`) | 分析摘要 |
| `src/engine/backtest/condition.py` | `compile_condition` (`src/engine/backtest/condition.py:22`) | 条件编译 |
| `src/engine/backtest/condition.py` | `evaluate_condition_series` (`src/engine/backtest/condition.py:125`) | 序列条件评估 |

#### 2.4.3 Execution / MarketData / Feature / PnL

| 模块 | 关键函数/类 | 说明 |
|---|---|---|
| `src/engine/execution/runtime_service.py` | `process_deployment_signal_cycle` (`src/engine/execution/runtime_service.py:213`) | 部署运行心跳主循环 |
| `src/engine/execution/signal_runtime.py` | `LiveSignalRuntime` (`src/engine/execution/signal_runtime.py:36`) | 在线信号决策 |
| `src/engine/execution/order_manager.py` | `OrderManager` (`src/engine/execution/order_manager.py:31`) | 下单状态机 |
| `src/engine/execution/risk_gate.py` | `RiskGate` (`src/engine/execution/risk_gate.py:32`) | 风险闸门 |
| `src/engine/execution/circuit_breaker.py` | `AsyncCircuitBreaker` (`src/engine/execution/circuit_breaker.py:27`) | 重试+熔断 |
| `src/engine/execution/kill_switch.py` | `RuntimeKillSwitch` (`src/engine/execution/kill_switch.py:28`) | 全局/用户/部署级熔断 |
| `src/engine/execution/adapters/alpaca_trading.py` | `AlpacaTradingAdapter` (`src/engine/execution/adapters/alpaca_trading.py:41`) | 券商适配器 |
| `src/engine/market_data/runtime.py` | `MarketDataRuntime` (`src/engine/market_data/runtime.py:55`) | 行情运行时缓存 |
| `src/engine/market_data/alpaca_client.py` | `AlpacaMarketDataClient` (`src/engine/market_data/alpaca_client.py:62`) | Alpaca 行情客户端 |
| `src/engine/market_data/aggregator.py` | `BarAggregator` (`src/engine/market_data/aggregator.py:81`) | 多周期聚合 |
| `src/engine/market_data/subscription_registry.py` | `SubscriptionRegistry` (`src/engine/market_data/subscription_registry.py:22`) | 订阅管理 |
| `src/engine/market_data/ring_buffer.py` | `OhlcvRing` (`src/engine/market_data/ring_buffer.py:11`) | 环形缓存 |
| `src/engine/feature/registry.py` | `FeatureRegistry` (`src/engine/feature/registry.py:11`) | 因子注册中心 |
| `src/engine/feature/indicators/registry.py` | `IndicatorRegistry` (`src/engine/feature/indicators/registry.py:14`) | 指标注册中心 |
| `src/engine/feature/indicators/wrapper.py` | `IndicatorWrapper` (`src/engine/feature/indicators/wrapper.py:42`) | 指标统一调用接口 |
| `src/engine/pnl/service.py` | `PnlService` (`src/engine/pnl/service.py:35`) | PnL 快照服务 |
| `src/engine/pnl/reconcile.py` | `reconcile_positions` (`src/engine/pnl/reconcile.py:25`) | 持仓对账 |

### 2.5 MCP 模块

| 模块 | 关键函数/类 | 说明 |
|---|---|---|
| `src/mcp/server.py` | `create_mcp_server` (`src/mcp/server.py:135`) | MCP server 装配入口 |
| `src/mcp/context_auth.py` | `create_mcp_context_token` (`src/mcp/context_auth.py:51`) | MCP 上下文 token 签发 |
| `src/mcp/context_auth.py` | `decode_mcp_context_token` (`src/mcp/context_auth.py:88`) | MCP 上下文 token 校验 |
| `src/mcp/dev_proxy.py` | `proxy_request` (`src/mcp/dev_proxy.py:171`) | 本地反向代理 |
| `src/mcp/strategy/tools.py` | `register_strategy_tools` (`src/mcp/strategy/tools.py:1633`) | strategy 域工具注册 |
| `src/mcp/backtest/tools.py` | `register_backtest_tools` (`src/mcp/backtest/tools.py:186`) | backtest 域工具注册 |
| `src/mcp/market_data/tools.py` | `register_market_data_tools` (`src/mcp/market_data/tools.py:970`) | market_data 域工具注册 |
| `src/mcp/stress/tools.py` | `stress_capabilities` (`src/mcp/stress/tools.py:44`) | stress 域能力探测（占位） |
| `src/mcp/trading/tools.py` | `trading_capabilities` (`src/mcp/trading/tools.py:44`) | trading 域能力探测（占位） |

### 2.6 Models / Services / Workers / Observability / Util

| 模块 | 关键函数/类 | 说明 |
|---|---|---|
| `src/models/database.py` | `init_postgres` (`src/models/database.py:144`) / `close_postgres` (`src/models/database.py:171`) | PG 连接生命周期 |
| `src/models/redis.py` | `init_redis` (`src/models/redis.py:17`) / `redis_healthcheck` (`src/models/redis.py:70`) | Redis 连接与健康检查 |
| `src/models/session.py` | `Session` (`src/models/session.py:31`), `Message` (`src/models/session.py:115`) | 会话与消息模型 |
| `src/models/user.py` | `User` (`src/models/user.py:28`), `UserProfile` (`src/models/user.py:90`) | 用户/KYC 模型 |
| `src/models/strategy.py` | `Strategy` (`src/models/strategy.py:33`) | 策略主表 |
| `src/models/strategy_revision.py` | `StrategyRevision` (`src/models/strategy_revision.py:26`) | 策略版本 |
| `src/models/backtest.py` | `BacktestJob` (`src/models/backtest.py:32`) | 回测任务模型 |
| `src/models/deployment.py` | `Deployment` (`src/models/deployment.py:34`) | 部署模型 |
| `src/models/deployment_run.py` | `DeploymentRun` (`src/models/deployment_run.py:21`) | 运行记录 |
| `src/models/order.py` | `Order` (`src/models/order.py:29`) | 订单模型 |
| `src/models/fill.py` | `Fill` (`src/models/fill.py:19`) | 成交模型 |
| `src/models/position.py` | `Position` (`src/models/position.py:18`) | 持仓模型 |
| `src/models/broker_account.py` | `BrokerAccount` (`src/models/broker_account.py:19`) | 券商账户 |
| `src/models/social_connector.py` | `SocialConnectorBinding` (`src/models/social_connector.py:30`) | 社交连接绑定 |
| `src/services/auth_service.py` | `AuthService` (`src/services/auth_service.py:29`) | 注册/登录/令牌刷新 |
| `src/services/openai_stream_service.py` | `ResponsesEventStreamer` (`src/services/openai_stream_service.py:317`) | OpenAI Responses 事件流封装 |
| `src/services/social_connector_service.py` | `SocialConnectorService` (`src/services/social_connector_service.py:47`) | Telegram connector 管理 |
| `src/services/telegram_service.py` | `TelegramService` (`src/services/telegram_service.py:17`) | Telegram webhook 处理 |
| `src/workers/backtest_tasks.py` | `execute_backtest_job_task` (`src/workers/backtest_tasks.py:35`) | 回测 worker 任务 |
| `src/workers/market_data_tasks.py` | `refresh_symbol_task` (`src/workers/market_data_tasks.py:32`) | 行情刷新任务 |
| `src/workers/paper_trading_tasks.py` | `run_deployment_runtime_task` (`src/workers/paper_trading_tasks.py:71`) | 运行时心跳任务 |
| `src/workers/maintenance_tasks.py` | `backup_postgres_full_task` (`src/workers/maintenance_tasks.py:201`) | PG 备份任务 |
| `src/observability/sentry_setup.py` | `init_backend_sentry` (`src/observability/sentry_setup.py:184`) | Sentry 初始化 |
| `src/observability/openai_cost.py` | `normalize_openai_usage` (`src/observability/openai_cost.py:59`) | Token 成本归一化 |
| `src/util/logger.py` | `configure_logging` (`src/util/logger.py:149`) | 日志总配置 |
| `src/util/data_setup.py` | `ensure_market_data` (`src/util/data_setup.py:97`) | 启动期数据准备 |
| `src/util/chat_debug_trace.py` | `build_chat_debug_trace` (`src/util/chat_debug_trace.py:114`) | 调试 trace 构建 |

## 3. 代码风格与工程约定摘要

### 3.1 命名/格式化
- 以 `ruff` 统一 lint 规则，启用 `N`（命名规范）、`I`（import 排序）等：`pyproject.toml:52`。
- 目标 Python 版本 `py312`，行宽 88，忽略 E501：`pyproject.toml:48`。
- 普遍使用类型注解与 `from __future__ import annotations`（如 `src/main.py:3`, `src/engine/backtest/service.py:3`）。

### 3.2 测试约定
- `pytest` 作为测试框架，测试目录固定 `tests`：`pyproject.toml:56`。
- `pytest-asyncio` 开启 `asyncio_mode=auto`：`pyproject.toml:58`。
- 测试组织按域拆分：`tests/test_api`, `tests/test_engine`, `tests/test_models`, `tests/test_mcp`, `tests/test_infra`, `tests/test_agents`。

### 3.3 异常处理约定
- API 层使用 `HTTPException` 返回结构化错误码（示例：`src/api/routers/backtests.py:68`, `src/api/routers/strategies.py:73`）。
- 引擎层定义领域异常（示例：`src/engine/backtest/service.py:60`, `src/engine/strategy/storage.py:91`）。
- 任务层对外部依赖故障使用降级/回滚（示例：`src/workers/maintenance_tasks.py:66`, `src/workers/maintenance_tasks.py:140`）。

### 3.4 依赖管理约定
- 单一依赖源 `pyproject.toml` + `uv sync --frozen`：`pyproject.toml:1`, `.github/workflows/deploy.yml:88`。
- Dev 依赖同样在 `pyproject.toml [tool.uv]` 管理：`pyproject.toml:40`。

## 4. 运行/部署/本地开发要点

### 4.1 常用命令

```bash
# 安装依赖
uv sync

# API
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# MCP（按域）
uv run python -m src.mcp.server --domain strategy --transport streamable-http --host 127.0.0.1 --port 8111
uv run python -m src.mcp.server --domain backtest --transport streamable-http --host 127.0.0.1 --port 8112
uv run python -m src.mcp.server --domain market --transport streamable-http --host 127.0.0.1 --port 8113

# 本地 MCP 代理
uv run python -m src.mcp.dev_proxy --host 127.0.0.1 --port 8110

# Celery worker / beat
uv run celery -A src.workers.celery_app.celery_app worker -l info -Q backtest
uv run celery -A src.workers.celery_app.celery_app beat -l info

# 测试 / Lint
uv run pytest
uv run ruff check .
```

### 4.2 环境与配置
- 变量模板：`.env.example`（覆盖 OpenAI、MCP、Alpaca、Postgres、Redis、Celery、CORS、Sentry）。
- 关键配置入口：`src/config.py:17`。
- 首启会尝试下载/解压 parquet 数据：`src/main.py:29`, `src/util/data_setup.py:97`。

### 4.3 部署与 CI/CD 入口
- CI/CD 入口：`.github/workflows/deploy.yml:1`（`push main` 自动 SSH 部署）。
- 服务器流程含：预部署备份、`uv sync --frozen`、systemd 重启、健康检查：`.github/workflows/deploy.yml:35`, `.github/workflows/deploy.yml:87`, `.github/workflows/deploy.yml:101`, `.github/workflows/deploy.yml:129`。
- 反向代理：`Caddyfile`（API + MCP 多域路由 + Flower）：`Caddyfile:1`, `Caddyfile:20`, `Caddyfile:74`。
- Flower systemd：`deploy/systemd/minsy-flower.service:10`。
- DB 恢复脚本：`scripts/restore_postgres_backup.sh:1`。

## 5. 未实现/技术债/风险点（含证据）

| 类型 | 说明 | 影响 | 证据 |
|---|---|---|---|
| 未实现 | stress MCP 域仅返回 `available=false` 占位能力 | 压测/优化工具链未接通 | `src/mcp/stress/tools.py:1`, `src/mcp/stress/tools.py:50` |
| 未实现 | trading MCP 域仅返回 `available=false` 占位能力 | 交易域 MCP 能力未接通 | `src/mcp/trading/tools.py:1`, `src/mcp/trading/tools.py:50` |
| 技术债 | strategy/stress 的停止条件为 placeholder（10 轮后提示） | 无真实绩效阈值，可能导致会话无效迭代 | `src/agents/orchestrator/fallback.py:9`, `src/agents/orchestrator/fallback.py:43` |
| 技术债 | 多个技术指标依赖 TA-Lib/pandas-ta，否则 `NotImplementedError` | 指标覆盖不完整，运行时可能失败 | `src/engine/feature/indicators/categories/momentum.py:136`, `src/engine/feature/indicators/categories/overlap.py:294`, `src/engine/feature/indicators/categories/volatility.py:257` |
| 技术债 | 行情流接口仍是 REST placeholder（未接 websocket stream） | 实时行情能力受限 | `src/engine/market_data/alpaca_client.py:257` |
| 设计约束 | 回测默认只取 `universe.tickers[0]` | 多标的策略回测被降维为单标的 | `src/engine/backtest/service.py:407` |
| 运行风险 | Docker CMD 指向 `api.main:app`，代码实际在 `src.main:app` | 容器启动路径不一致可能导致启动失败 | `Dockerfile:35`, `src/main.py:65` |
| 运行风险 | 应用生命周期同步执行数据下载解压 | 冷启动耗时 + 外网/7z 依赖 | `src/main.py:29`, `src/util/data_setup.py:30`, `src/util/data_setup.py:39` |
| 运维风险 | 部署脚本使用 `git reset --hard origin/main` | 服务器本地改动会被强制覆盖 | `.github/workflows/deploy.yml:80` |

## 6. 推荐索引/检索策略（简短）

1. 优先符号索引：先查 `docs/backend_symbol_index.json`，按 `path/module/symbol/line` 直接定位。  
2. 再做全文搜索：用 `rg "关键词|符号名" backend/src backend/tests backend/scripts` 补上下文。  
3. 不做向量索引：当前代码库更适合“结构化符号 + 关键词 grep”，可复现、可审计、低维护。

## 7. 输出格式与导航约定

- 文档导航：本 README 使用固定 TOC + 分模块表格。
- 路径定位：统一 `path:line`（例如 `src/engine/backtest/service.py:167`）。
- 机器可读元数据：
  - 文档头部 YAML front matter（生成时间、范围、统计、检索策略）。
  - 全量符号索引：`docs/backend_symbol_index.json`。
- 索引数据文件位置：`backend/docs/backend_symbol_index.json`（仓库内相对路径：`docs/backend_symbol_index.json`）。

