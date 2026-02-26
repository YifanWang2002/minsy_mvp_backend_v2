# Main 分支历史测试能力清单与测试方法

说明: 本文档通过读取 `main` worktree (`.codex_tmp/backend-main/tests`) 生成，不复制旧测试源码。

| 测试文件 | 能力项目 | 主要测试方法 |
|---|---|---|
| `test_agents/test_genui_registry.py` | genui registry | 流式事件序列断言 |
| `test_agents/test_orchestrator_runtime_policy.py` | orchestrator runtime policy | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_agents/test_phase_handlers.py` | phase handlers | 流式事件序列断言 |
| `test_agents/test_pre_strategy_dynamic_rules.py` | pre strategy dynamic rules | 依赖注入与错误路径验证; 流式事件序列断言; 规则/语义校验 |
| `test_agents/test_session_title_service.py` | session title service | 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_agents/test_static_instruction_cache.py` | static instruction cache | 流式事件序列断言; 规则/语义校验 |
| `test_agents/test_stub_handler_stage_loading.py` | stub handler stage loading | 流式事件序列断言; 规则/语义校验 |
| `test_api/test_auth_flow_from_script.py` | auth flow from script | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_broker_accounts_credentials.py` | broker accounts credentials | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_broker_accounts_paper_only.py` | broker accounts paper only | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_change_password.py` | change password | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_chat_debug_trace_header.py` | chat debug trace header | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_chat_stream.py` | chat stream | API 合约/鉴权/状态码; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_chat_stream_transport_resilience.py` | chat stream transport resilience | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_cors_mode.py` | cors mode | 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_api/test_deployment_signal_order_flow.py` | deployment signal order flow | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_deployments_lifecycle.py` | deployments lifecycle | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_health.py` | health | API 合约/鉴权/状态码; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_jwt.py` | jwt | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_kyc_e2e_verbose.py` | kyc e2e verbose | API 合约/鉴权/状态码; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_kyc_flow.py` | kyc flow | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_kyc_flow_from_script.py` | kyc flow from script | API 合约/鉴权/状态码; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_login.py` | login | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_login_response.py` | login response | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_manual_trade_actions.py` | manual trade actions | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_market_data_endpoints.py` | market data endpoints | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_api/test_mcp_history_persistence.py` | mcp history persistence | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_me.py` | me | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_notification_preferences.py` | notification preferences | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_paper_trading_full_pipeline.py` | paper trading full pipeline | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_portfolio_endpoints.py` | portfolio endpoints | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_pre_strategy_robustness.py` | pre strategy robustness | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_pre_strategy_to_strategy_e2e_live.py` | pre strategy to strategy e2e live | API 合约/鉴权/状态码; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_preferences.py` | preferences | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_rate_limit.py` | rate limit | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_refresh.py` | refresh | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_register.py` | register | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_session_titles.py` | session titles | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_sessions_archive.py` | sessions archive | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_api/test_social_connectors.py` | social connectors | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_startup_shutdown.py` | startup shutdown | API 合约/鉴权/状态码; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_status.py` | status | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_api/test_status_live_trading_probe.py` | status live trading probe | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_strategies_list_versions.py` | strategies list versions | API 合约/鉴权/状态码; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_strategy_confirm.py` | strategy confirm | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_strategy_stream_live.py` | strategy stream live | API 合约/鉴权/状态码; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言 |
| `test_api/test_stream_empty_output_fallback.py` | stream empty output fallback | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_stream_error_detail.py` | stream error detail | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_api/test_telegram_test_target_endpoint.py` | telegram test target endpoint | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_api/test_telegram_trade_approval_callback.py` | telegram trade approval callback | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_trade_approvals_flow.py` | trade approvals flow | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_api/test_trading_preferences.py` | trading preferences | API 合约/鉴权/状态码; 流式事件序列断言 |
| `test_engine/test_alpaca_account_probe.py` | alpaca account probe | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_engine/test_alpaca_market_data_client_symbol_normalization.py` | alpaca market data client symbol normalization | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_engine/test_alpaca_order_sync.py` | alpaca order sync | 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_alpaca_quote_endpoint_fallback.py` | alpaca quote endpoint fallback | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_engine/test_alpaca_stream_reconnect.py` | alpaca stream reconnect | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_engine/test_backtest_analytics.py` | backtest analytics | 流式事件序列断言 |
| `test_engine/test_backtest_condition_vectorized.py` | backtest condition vectorized | 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_backtest_engine.py` | backtest engine | 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_backtest_service.py` | backtest service | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_bar_aggregator_boundaries.py` | bar aggregator boundaries | 流式事件序列断言 |
| `test_engine/test_black_swan_scenarios.py` | black swan scenarios | 流式事件序列断言 |
| `test_engine/test_broker_adapter_contract.py` | broker adapter contract | 流式事件序列断言 |
| `test_engine/test_credentials_cipher.py` | credentials cipher | 流式事件序列断言 |
| `test_engine/test_data_loader.py` | data loader | 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_engine/test_factor_cache_dedup.py` | factor cache dedup | 流式事件序列断言 |
| `test_engine/test_feature_indicator_wrapper.py` | feature indicator wrapper | 流式事件序列断言 |
| `test_engine/test_feature_registry.py` | feature registry | 流式事件序列断言 |
| `test_engine/test_live_dsl_signal_runtime.py` | live dsl signal runtime | 流式事件序列断言 |
| `test_engine/test_local_coverage_detector.py` | local coverage detector | 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_engine/test_market_data_redis_store.py` | market data redis store | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_engine/test_market_data_sync_service.py` | market data sync service | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言 |
| `test_engine/test_monte_carlo_statistics.py` | monte carlo statistics | 流式事件序列断言 |
| `test_engine/test_optimization_algorithms.py` | optimization algorithms | 流式事件序列断言 |
| `test_engine/test_optimization_pareto.py` | optimization pareto | 流式事件序列断言 |
| `test_engine/test_order_idempotency.py` | order idempotency | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_order_state_machine.py` | order state machine | 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_engine/test_param_mutation.py` | param mutation | 流式事件序列断言 |
| `test_engine/test_param_sensitivity_scan.py` | param sensitivity scan | 流式事件序列断言 |
| `test_engine/test_performance_quantstats.py` | performance quantstats | 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_pnl_engine.py` | pnl engine | 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_position_reconcile.py` | position reconcile | 流式事件序列断言 |
| `test_engine/test_ring_buffer.py` | ring buffer | 流式事件序列断言 |
| `test_engine/test_risk_gate.py` | risk gate | 流式事件序列断言 |
| `test_engine/test_strategy_dsl_pipeline.py` | strategy dsl pipeline | 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_strategy_storage.py` | strategy storage | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_stress_job_service.py` | stress job service | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_engine/test_timeframe_scheduler.py` | timeframe scheduler | 流式事件序列断言 |
| `test_infra/test_backtest_tasks.py` | backtest tasks | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_broker_retry_and_circuit_breaker.py` | broker retry and circuit breaker | 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_infra/test_celery_limits.py` | celery limits | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_celery_paper_trading_queue.py` | celery paper trading queue | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_celery_trade_approval_queue.py` | celery trade approval queue | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_chat_debug_trace.py` | chat debug trace | 流式事件序列断言 |
| `test_infra/test_config.py` | config | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_infra/test_db_connection.py` | db connection | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_deployment_runtime_lock.py` | deployment runtime lock | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_maintenance_tasks.py` | maintenance tasks | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言 |
| `test_infra/test_market_data_refresh_dedupe.py` | market data refresh dedupe | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_market_data_tasks_redis_dual_write.py` | market data tasks redis dual write | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_mcp_context_auth.py` | mcp context auth | 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_infra/test_openai_cost_tracker.py` | openai cost tracker | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_openai_stream_service_errors.py` | openai stream service errors | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_redis_connection.py` | redis connection | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_runtime_kill_switch.py` | runtime kill switch | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言; 规则/语义校验 |
| `test_infra/test_runtime_state_store.py` | runtime state store | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_sentry_http_status_middleware.py` | sentry http status middleware | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_infra/test_sentry_integration_bootstrap.py` | sentry integration bootstrap | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_sentry_setup.py` | sentry setup | API 合约/鉴权/状态码; 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_infra/test_signal_store_externalization.py` | signal store externalization | 外部依赖/集成链路验证; 流式事件序列断言 |
| `test_mcp/test_backtest_tools.py` | backtest tools | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_mcp/test_dev_proxy.py` | dev proxy | API 合约/鉴权/状态码; 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_mcp/test_market_data_sync_tools.py` | market data sync tools | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言 |
| `test_mcp/test_market_data_tools.py` | market data tools | 依赖注入与错误路径验证; 流式事件序列断言 |
| `test_mcp/test_market_data_tools_live.py` | market data tools live | 流式事件序列断言 |
| `test_mcp/test_server_domains.py` | server domains | 依赖注入与错误路径验证; 流式事件序列断言; 规则/语义校验 |
| `test_mcp/test_strategy_tools.py` | strategy tools | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_mcp/test_stress_tools.py` | stress tools | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言 |
| `test_mcp/test_trading_tools.py` | trading tools | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_models/test_auth_service.py` | auth service | 数据库状态断言; 流式事件序列断言 |
| `test_models/test_constraints.py` | constraints | 依赖注入与错误路径验证; 数据库状态断言 |
| `test_models/test_jsonb_fields.py` | jsonb fields | 数据库状态断言; 流式事件序列断言 |
| `test_models/test_models.py` | models | 数据库状态断言; 流式事件序列断言 |
| `test_models/test_relationships.py` | relationships | 数据库状态断言; 流式事件序列断言 |
| `test_models/test_seed_data_from_script.py` | seed data from script | 数据库状态断言; 流式事件序列断言 |
| `test_models/test_trade_approval_request_model.py` | trade approval request model | 依赖注入与错误路径验证; 数据库状态断言; 规则/语义校验 |
| `test_models/test_trading_models.py` | trading models | 依赖注入与错误路径验证; 外部依赖/集成链路验证; 数据库状态断言; 流式事件序列断言; 规则/语义校验 |
| `test_models/test_user_settings.py` | user settings | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言 |
| `test_services/test_notification_outbox_service.py` | notification outbox service | 依赖注入与错误路径验证; 数据库状态断言; 流式事件序列断言 |
