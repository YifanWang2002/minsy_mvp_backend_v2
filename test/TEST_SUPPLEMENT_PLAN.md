# Backend 测试补充计划与验收标准（Real-Dependency）

## 1. 现有测试覆盖清单（已完成）

### 1.1 `test/db/postgres`
- 文件: `test_postgres_live.py`
- 已覆盖:
1. 本地 PostgreSQL `127.0.0.1:5432` 可连通。
2. 种子用户 `2@test.com` 存在。
3. 种子用户明文密码 `123456` 与 `password_hash` 匹配。

### 1.2 `test/db/parquet`
- 文件: `test_parquet_loader_live.py`
- 已覆盖:
1. 本地 `data/**/*.parquet` 存在。
2. `DataLoader.load` 可真实加载 `crypto/BTCUSD` 区间数据。
3. `DataLoader.get_symbol_metadata` 可返回 `stocks/SPY` 元数据。

### 1.3 `test/api/auth`
- 文件: `test_auth_live.py`
- 已覆盖:
1. `POST /api/v1/auth/login`（真实 DB 账号）成功。
2. `GET /api/v1/auth/me` 返回账号身份。
3. `POST /api/v1/auth/refresh` 返回新 token。

### 1.4 `test/api/chat`
- 文件: `test_openai_stream_live.py`
- 已覆盖:
1. `POST /api/v1/chat/send-openai-stream` 真实 OpenAI Responses 流调用。
2. `response.created` / `response.completed` 事件链。
3. `done.usage.total_tokens > 0`。
4. 文本完成度（至少 3 条要点）。

### 1.5 `test/api/market_data`
- 文件: `test_market_data_live.py`
- 已覆盖:
1. `GET /api/v1/market-data/quote` 真实 Alpaca quote 刷新。
2. `POST/GET/DELETE /api/v1/market-data/subscriptions` 生命周期。
3. `GET /api/v1/market-data/health` 结构字段。

### 1.6 `test/mcp/router`
- 文件: `test_mcp_router_live.py`
- 已覆盖:
1. 本地 MCP Router 各 domain 路径可达。
2. 对各 domain 执行 `tools/list`，返回真实工具列表。
3. `.env.dev` 中公网 MCP URL 配置存在且合法（`https://.../mcp`）。

### 1.7 `test/celery/worker_io`
- 文件: `test_worker_io_live.py`
- 已覆盖:
1. `celery inspect active_queues` 能看到 IO/CPU worker 队列。
2. `market_data.refresh_symbol` 真任务执行。
3. Alpaca paper 真下单（提交+必要清理）。

### 1.8 `test/containers/compose`
- 文件: `test_compose_stack_live.py`
- 已覆盖:
1. `compose.dev.yml` 七个服务均 `running`。
2. API/MCP 对外端点可达。
3. 关键服务日志无致命错误标记。

## 2. 现状缺口
- 每个二级子目录目前基本只有 1 个测试文件，横向功能覆盖不足。
- 对 API 的 session/chat/thread、auth 偏好、market data checkpoint 等场景覆盖不足。
- 对 Celery IO 多任务（backfill/scheduler 等）覆盖不足。
- 对 MCP 的 `tools/call`（不止 `tools/list`）覆盖不足。
- 对容器层的 worker 在线性与队列拓扑覆盖不足。
- 对 parquet 的市场/时间框架/元数据一致性覆盖不足。
- 对 DB 的 schema/计数/关键业务表访问覆盖不足。

## 3. 补充测试计划（本轮新增）

### 3.1 `test/db/postgres` 新增
1. `test_postgres_schema_live.py`
- 校验关键业务表存在（sessions/strategies/deployments/orders/positions/...）。
- 校验核心表行数查询可执行（非空约束以“可查询”为主）。

2. `test_postgres_user_state_live.py`
- 校验种子用户可关联查询 profile/settings/session 相关数据。
- 校验用户主键在关键业务表外键查询路径可执行。

### 3.2 `test/db/parquet` 新增
1. `test_parquet_market_inventory_live.py`
- 校验四大 market 目录存在可读 parquet。
- 校验 `get_available_markets/get_available_symbols` 返回非空。

2. `test_parquet_timeframe_live.py`
- 校验 `1m->5m/15m` 重采样可执行并返回有效数据。
- 校验不同 market（stocks/crypto）读取路径。

### 3.3 `test/api/auth` 新增
1. `test_auth_preferences_live.py`
- `GET /auth/preferences`。
- `PUT /auth/preferences` 更新并回读一致。

2. `test_auth_negative_live.py`
- 错误密码登录失败路径。
- 缺失 token 调用受保护接口返回 401。

### 3.4 `test/api/chat` 新增
1. `test_chat_thread_session_live.py`
- `POST /chat/new-thread` 创建会话。
- `GET /sessions` 能看到会话。
- `GET /sessions/{id}` 明细可读。

2. `test_chat_session_archive_live.py`
- `archive -> list(archived=true) -> unarchive` 流程。

### 3.5 `test/api/market_data` 新增
1. `test_market_data_bars_checkpoint_live.py`
- `GET /market-data/checkpoints`。
- `GET /market-data/bars`（有缓存/空结果都要走真实接口，断言结构）。

2. `test_market_data_validation_live.py`
- 非法 market 参数返回 422。
- quote 未命中逻辑下错误码结构校验。

### 3.6 `test/mcp/router` 新增
1. `test_mcp_tools_call_live.py`
- 对无上下文依赖的工具执行 `tools/call`：
  - strategy: `get_indicator_catalog`
  - stress: `stress_ping`
  - trading: `trading_ping`
  - market: `check_symbol_available`

2. `test_mcp_router_negative_live.py`
- 错误路由返回 404。
- 错误 `Accept` 头返回 4xx 且结构合理。

### 3.7 `test/celery/worker_io` 新增
1. `test_worker_io_tasks_live.py`
- `market_data.backfill_symbol` 真执行。
- `market_data.refresh_active_subscriptions` 真执行返回结构。

2. `test_worker_io_runtime_probe_live.py`
- `celery inspect stats/ping` 至少返回一节点。
- worker 命名和队列绑定符合预期。

### 3.8 `test/containers/compose` 新增
1. `test_compose_worker_probe_live.py`
- 在容器内执行 celery inspect，验证 worker-cpu/worker-io 在线。

2. `test_compose_restart_stability_live.py`
- 对单服务（如 `mcp`）执行重启并验证恢复可用。

## 4. 验收标准

### 4.1 全局验收标准
1. 禁止 `skip/xfail`；依赖不可用时必须失败并给出可诊断信息。
2. 仅使用真实依赖：
- 本地 PostgreSQL(`5432`)；
- 本地 parquet；
- 真实 OpenAI Responses；
- 真实 Alpaca 行情/paper 交易；
- 真实 MCP domain router；
- 真实 Docker Compose 多容器。
3. 每个二级子目录至少 2 个测试文件。
4. `uv run pytest -q` 全量通过。

### 4.2 API 相关验收
1. 对每个 API 子目录，至少包含：
- accessibility 测试（第一个用例，命名 `test_000_accessibility_*`）；
- 功能正向；
- 负向/边界。

### 4.3 任务与容器验收
1. Celery 至少覆盖 2 类真实任务运行。
2. Compose 至少覆盖：服务在线、端点可达、worker 在线、重启恢复。

### 4.4 数据验收
1. Postgres 至少覆盖：连通、用户、schema/关键表访问。
2. Parquet 至少覆盖：库存、读取、元数据、重采样。

## 5. 执行顺序
1. 先补每个子目录第二个测试文件。
2. 再补负向/边界用例。
3. 全量跑测并修复波动点（重试/等待策略仅用于基础设施抖动，不掩盖逻辑错误）。
