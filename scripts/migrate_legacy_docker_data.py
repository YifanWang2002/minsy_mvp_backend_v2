#!/usr/bin/env python3
"""Migrate legacy standalone Docker Postgres/Redis data into compose services."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

import redis
from redis.exceptions import RedisError

SOURCE_CLEANUP_CONFIRM_TOKEN = "DELETE_SOURCE"


def parse_db_list(raw: str) -> list[int]:
    dbs: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            db = int(stripped)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"Invalid redis db index: {stripped}") from exc
        if db < 0:
            raise ValueError(f"Redis db index must be >= 0: {db}")
        dbs.append(db)
    unique_sorted = sorted(set(dbs))
    if not unique_sorted:
        raise ValueError("At least one redis db must be provided.")
    return unique_sorted


def ensure_docker_container_running(container_name: str) -> None:
    cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Container not found: {container_name}. "
            f"docker inspect failed: {result.stderr.strip()}"
        )
    if result.stdout.strip().lower() != "true":
        raise RuntimeError(f"Container is not running: {container_name}")


def run_psql_scalar(
    *,
    container_name: str,
    pg_user: str,
    pg_password: str,
    pg_db: str,
    sql: str,
) -> str:
    cmd = [
        "docker",
        "exec",
        "-e",
        f"PGPASSWORD={pg_password}",
        container_name,
        "psql",
        "-U",
        pg_user,
        "-d",
        pg_db,
        "-At",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"psql failed in container={container_name}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def list_user_tables(
    *,
    container_name: str,
    pg_user: str,
    pg_password: str,
    pg_db: str,
) -> list[str]:
    output = run_psql_scalar(
        container_name=container_name,
        pg_user=pg_user,
        pg_password=pg_password,
        pg_db=pg_db,
        sql=(
            "SELECT format('%I.%I', schemaname, tablename) "
            "FROM pg_tables "
            "WHERE schemaname NOT IN ('pg_catalog','information_schema') "
            "ORDER BY 1;"
        ),
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def collect_postgres_table_counts(
    *,
    container_name: str,
    pg_user: str,
    pg_password: str,
    pg_db: str,
) -> dict[str, int]:
    ensure_docker_container_running(container_name)
    counts: dict[str, int] = {}
    for table_name in list_user_tables(
        container_name=container_name,
        pg_user=pg_user,
        pg_password=pg_password,
        pg_db=pg_db,
    ):
        count_raw = run_psql_scalar(
            container_name=container_name,
            pg_user=pg_user,
            pg_password=pg_password,
            pg_db=pg_db,
            sql=f"SELECT count(*)::bigint FROM {table_name};",
        )
        counts[table_name] = int(count_raw or "0")
    return counts


def verify_postgres_counts(args: argparse.Namespace) -> tuple[bool, list[str]]:
    print("[verify][postgres] comparing per-table row counts...")
    src_counts = collect_postgres_table_counts(
        container_name=args.src_pg_container,
        pg_user=args.src_pg_user,
        pg_password=args.src_pg_password,
        pg_db=args.src_pg_db,
    )
    dst_counts = collect_postgres_table_counts(
        container_name=args.dst_pg_container,
        pg_user=args.dst_pg_user,
        pg_password=args.dst_pg_password,
        pg_db=args.dst_pg_db,
    )

    issues: list[str] = []
    for table_name in sorted(src_counts):
        if table_name not in dst_counts:
            issues.append(f"missing table in destination: {table_name}")
            continue
        if src_counts[table_name] != dst_counts[table_name]:
            issues.append(
                f"row mismatch {table_name}: "
                f"src={src_counts[table_name]} dst={dst_counts[table_name]}"
            )
    for table_name in sorted(dst_counts):
        if table_name not in src_counts:
            issues.append(f"extra table in destination: {table_name}")

    src_total_rows = sum(src_counts.values())
    dst_total_rows = sum(dst_counts.values())
    print(
        "[verify][postgres] totals "
        f"src_tables={len(src_counts)} src_rows={src_total_rows} "
        f"dst_tables={len(dst_counts)} dst_rows={dst_total_rows}"
    )
    if issues:
        print(f"[verify][postgres] failed, issues={len(issues)}")
        for issue in issues:
            print(f"[verify][postgres] issue: {issue}")
        return False, issues
    print("[verify][postgres] passed")
    return True, []


def migrate_postgres(args: argparse.Namespace) -> None:
    ensure_docker_container_running(args.src_pg_container)
    ensure_docker_container_running(args.dst_pg_container)

    dump_cmd = [
        "docker",
        "exec",
        "-e",
        f"PGPASSWORD={args.src_pg_password}",
        args.src_pg_container,
        "pg_dump",
        "-U",
        args.src_pg_user,
        "-d",
        args.src_pg_db,
        "--format=plain",
        "--encoding=UTF8",
        "--no-owner",
        "--no-privileges",
    ]
    if not args.pg_no_clean:
        dump_cmd.extend(["--clean", "--if-exists"])

    restore_cmd = [
        "docker",
        "exec",
        "-i",
        "-e",
        f"PGPASSWORD={args.dst_pg_password}",
        args.dst_pg_container,
        "psql",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        args.dst_pg_user,
        "-d",
        args.dst_pg_db,
    ]

    print(
        "[postgres] migrate "
        f"{args.src_pg_container}:{args.src_pg_db} -> "
        f"{args.dst_pg_container}:{args.dst_pg_db}"
    )

    dump_proc = subprocess.Popen(dump_cmd, stdout=subprocess.PIPE)
    if dump_proc.stdout is None:  # pragma: no cover - defensive check
        raise RuntimeError("Failed to open pg_dump stdout.")

    restore_proc = subprocess.Popen(restore_cmd, stdin=dump_proc.stdout)
    dump_proc.stdout.close()

    restore_code = restore_proc.wait()
    dump_code = dump_proc.wait()

    if dump_code != 0:
        raise RuntimeError(f"pg_dump failed with exit code {dump_code}.")
    if restore_code != 0:
        raise RuntimeError(f"psql restore failed with exit code {restore_code}.")

    print("[postgres] done")


def build_redis_client(
    *,
    host: str,
    port: int,
    db: int,
    password: str,
) -> redis.Redis:
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password or None,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=30,
    )


@dataclass(slots=True)
class RedisVerificationResult:
    db: int
    source_keys: int
    destination_keys: int
    missing_in_destination: int
    value_mismatch: int
    extra_in_destination: int

    @property
    def passed(self) -> bool:
        return self.missing_in_destination == 0 and self.value_mismatch == 0


def verify_redis_db(
    args: argparse.Namespace,
    db: int,
) -> RedisVerificationResult:
    src = build_redis_client(
        host=args.src_redis_host,
        port=args.src_redis_port,
        db=db,
        password=args.src_redis_password,
    )
    dst = build_redis_client(
        host=args.dst_redis_host,
        port=args.dst_redis_port,
        db=db,
        password=args.dst_redis_password,
    )

    try:
        src.ping()
        dst.ping()
    except RedisError as exc:
        raise RuntimeError(f"Redis ping failed for verify db {db}: {exc}") from exc

    source_keys = sorted(
        src.scan_iter(match=args.redis_match, count=args.redis_scan_count)
    )
    destination_key_set = set(
        dst.scan_iter(match=args.redis_match, count=args.redis_scan_count)
    )

    missing_in_destination = 0
    value_mismatch = 0
    source_key_set = set(source_keys)
    for key in source_keys:
        if key not in destination_key_set:
            missing_in_destination += 1
            continue
        if src.dump(key) != dst.dump(key):
            value_mismatch += 1

    extra_in_destination = 0
    for key in destination_key_set:
        if key not in source_key_set:
            extra_in_destination += 1

    return RedisVerificationResult(
        db=db,
        source_keys=len(source_keys),
        destination_keys=len(destination_key_set),
        missing_in_destination=missing_in_destination,
        value_mismatch=value_mismatch,
        extra_in_destination=extra_in_destination,
    )


def migrate_redis_db(args: argparse.Namespace, db: int) -> int:
    src = build_redis_client(
        host=args.src_redis_host,
        port=args.src_redis_port,
        db=db,
        password=args.src_redis_password,
    )
    dst = build_redis_client(
        host=args.dst_redis_host,
        port=args.dst_redis_port,
        db=db,
        password=args.dst_redis_password,
    )

    try:
        src.ping()
        dst.ping()
    except RedisError as exc:
        raise RuntimeError(f"Redis ping failed for db {db}: {exc}") from exc

    print(
        f"[redis][db={db}] migrate "
        f"{args.src_redis_host}:{args.src_redis_port} -> "
        f"{args.dst_redis_host}:{args.dst_redis_port}"
    )

    if args.flush_target_redis:
        print(f"[redis][db={db}] flush destination db before restore")
        dst.flushdb()

    total_keys = 0
    cursor = 0
    while True:
        cursor, keys = src.scan(
            cursor=cursor,
            match=args.redis_match,
            count=args.redis_scan_count,
        )

        if keys:
            pipeline = dst.pipeline(transaction=False)
            restoring_keys: list[bytes] = []

            for key in keys:
                payload = src.dump(key)
                if payload is None:
                    continue
                ttl_ms = src.pttl(key)
                if ttl_ms < 0:
                    ttl_ms = 0
                pipeline.restore(key, ttl_ms, payload, replace=True)
                restoring_keys.append(key)

            if restoring_keys:
                results = pipeline.execute(raise_on_error=False)
                for key, result in zip(restoring_keys, results):
                    if isinstance(result, Exception):
                        key_preview = key.decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Failed to restore redis key '{key_preview}' "
                            f"in db {db}: {result}"
                        )
                total_keys += len(restoring_keys)
                if total_keys % args.redis_progress_every < len(restoring_keys):
                    print(f"[redis][db={db}] migrated {total_keys} keys...")

        if cursor == 0:
            break

    print(f"[redis][db={db}] done, migrated {total_keys} keys")
    return total_keys


def verify_redis(args: argparse.Namespace, redis_dbs: list[int]) -> tuple[bool, list[str]]:
    print("[verify][redis] checking key existence and serialized value parity...")
    issues: list[str] = []
    for db in redis_dbs:
        result = verify_redis_db(args, db)
        print(
            f"[verify][redis][db={db}] "
            f"src={result.source_keys} dst={result.destination_keys} "
            f"missing={result.missing_in_destination} "
            f"value_mismatch={result.value_mismatch} "
            f"extra={result.extra_in_destination}"
        )
        if result.missing_in_destination > 0:
            issues.append(
                f"db{db}: missing_in_destination={result.missing_in_destination}"
            )
        if result.value_mismatch > 0:
            issues.append(f"db{db}: value_mismatch={result.value_mismatch}")
        if args.fail_on_extra_redis_keys and result.extra_in_destination > 0:
            issues.append(f"db{db}: extra_in_destination={result.extra_in_destination}")

    if issues:
        print(f"[verify][redis] failed, issues={len(issues)}")
        for issue in issues:
            print(f"[verify][redis] issue: {issue}")
        return False, issues
    print("[verify][redis] passed")
    return True, []


def cleanup_source_postgres(args: argparse.Namespace) -> None:
    ensure_docker_container_running(args.src_pg_container)
    print(
        "[cleanup][postgres] verified copy completed, "
        "dropping source public schema..."
    )
    run_psql_scalar(
        container_name=args.src_pg_container,
        pg_user=args.src_pg_user,
        pg_password=args.src_pg_password,
        pg_db=args.src_pg_db,
        sql=(
            "DROP SCHEMA IF EXISTS public CASCADE; "
            "CREATE SCHEMA public; "
            f"GRANT ALL ON SCHEMA public TO {args.src_pg_user}; "
            "GRANT ALL ON SCHEMA public TO public;"
        ),
    )
    print("[cleanup][postgres] source schema reset completed")


def cleanup_source_redis(args: argparse.Namespace, redis_dbs: list[int]) -> None:
    for db in redis_dbs:
        client = build_redis_client(
            host=args.src_redis_host,
            port=args.src_redis_port,
            db=db,
            password=args.src_redis_password,
        )
        client.ping()
        deleted = client.dbsize()
        client.flushdb()
        print(f"[cleanup][redis][db={db}] flushed keys={deleted}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy standalone docker postgres/redis data "
            "to current compose postgres/redis."
        )
    )

    parser.add_argument("--skip-postgres", action="store_true")
    parser.add_argument("--skip-redis", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip migration and only run source/destination verification.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification checks after migration.",
    )
    parser.add_argument(
        "--cleanup-source-after-verify",
        action="store_true",
        help="Delete source postgres/redis data only after verification passes.",
    )
    parser.add_argument(
        "--confirm-source-cleanup",
        default="",
        help=(
            "Required token when using --cleanup-source-after-verify. "
            f"Must be exactly: {SOURCE_CLEANUP_CONFIRM_TOKEN}"
        ),
    )
    parser.add_argument(
        "--fail-on-extra-redis-keys",
        action="store_true",
        help="Treat extra keys in destination redis as verification failure.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )

    parser.add_argument("--src-pg-container", default=os.getenv("SRC_PG_CONTAINER", "postgres"))
    parser.add_argument(
        "--dst-pg-container",
        default=os.getenv("DST_PG_CONTAINER", "minsy-postgres-dev"),
    )
    parser.add_argument("--src-pg-db", default=os.getenv("SRC_PG_DB", "minsy_pgsql"))
    parser.add_argument("--dst-pg-db", default=os.getenv("DST_PG_DB", "minsy_pgsql"))
    parser.add_argument("--src-pg-user", default=os.getenv("SRC_PG_USER", "postgres"))
    parser.add_argument("--dst-pg-user", default=os.getenv("DST_PG_USER", "postgres"))
    parser.add_argument("--src-pg-password", default=os.getenv("SRC_PG_PASSWORD", "123456"))
    parser.add_argument("--dst-pg-password", default=os.getenv("DST_PG_PASSWORD", "123456"))
    parser.add_argument(
        "--pg-no-clean",
        action="store_true",
        help="Do not drop existing objects in destination before restore.",
    )

    parser.add_argument("--src-redis-host", default=os.getenv("SRC_REDIS_HOST", "127.0.0.1"))
    parser.add_argument(
        "--dst-redis-host",
        default=os.getenv("DST_REDIS_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--src-redis-port",
        type=int,
        default=int(os.getenv("SRC_REDIS_PORT", "6379")),
    )
    parser.add_argument(
        "--dst-redis-port",
        type=int,
        default=int(os.getenv("DST_REDIS_PORT", "6380")),
    )
    parser.add_argument(
        "--src-redis-password",
        default=os.getenv("SRC_REDIS_PASSWORD", ""),
    )
    parser.add_argument(
        "--dst-redis-password",
        default=os.getenv("DST_REDIS_PASSWORD", ""),
    )
    parser.add_argument(
        "--redis-dbs",
        default=os.getenv("REDIS_DBS", "0"),
        help="Comma-separated redis db indexes. Example: 0,1,2",
    )
    parser.add_argument("--redis-match", default=os.getenv("REDIS_MATCH", "*"))
    parser.add_argument(
        "--redis-scan-count",
        type=int,
        default=int(os.getenv("REDIS_SCAN_COUNT", "500")),
    )
    parser.add_argument(
        "--redis-progress-every",
        type=int,
        default=int(os.getenv("REDIS_PROGRESS_EVERY", "1000")),
    )
    parser.add_argument(
        "--flush-target-redis",
        action="store_true",
        help="Flush each destination redis db before restore.",
    )

    return parser


def confirm_or_abort(args: argparse.Namespace, redis_dbs: list[int]) -> None:
    if args.yes:
        return
    print("Migration plan:")
    if not args.skip_postgres:
        print(
            "  Postgres: "
            f"{args.src_pg_container}:{args.src_pg_db} -> "
            f"{args.dst_pg_container}:{args.dst_pg_db}"
        )
    if not args.skip_redis:
        print(
            "  Redis: "
            f"{args.src_redis_host}:{args.src_redis_port} -> "
            f"{args.dst_redis_host}:{args.dst_redis_port}, dbs={redis_dbs}"
        )
    print(f"  Verify after migrate: {not args.skip_verify}")
    print(
        "  Cleanup source after verify: "
        f"{args.cleanup_source_after_verify} "
        "(copy-first; source remains intact unless cleanup is explicitly enabled)"
    )
    if args.verify_only:
        print("  Mode: verify-only (no data copy).")
    print("Destination data may be overwritten.")
    answer = input("Continue? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise RuntimeError("Aborted by user.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.skip_postgres and args.skip_redis:
        parser.error("Nothing to do. Remove --skip-postgres or --skip-redis.")
    if args.verify_only and args.skip_verify:
        parser.error("--verify-only cannot be used together with --skip-verify.")
    if args.cleanup_source_after_verify and args.skip_verify:
        parser.error("--cleanup-source-after-verify requires verification.")
    if args.cleanup_source_after_verify and (
        args.confirm_source_cleanup != SOURCE_CLEANUP_CONFIRM_TOKEN
    ):
        parser.error(
            "--cleanup-source-after-verify requires "
            f"--confirm-source-cleanup {SOURCE_CLEANUP_CONFIRM_TOKEN}"
        )

    try:
        redis_dbs = parse_db_list(args.redis_dbs)
        confirm_or_abort(args, redis_dbs)

        if not args.verify_only:
            if not args.skip_postgres:
                migrate_postgres(args)

            if not args.skip_redis:
                total = 0
                for db in redis_dbs:
                    total += migrate_redis_db(args, db)
                print(f"[redis] total migrated keys: {total}")

        verified = True
        if args.skip_verify:
            print("[verify] skipped by --skip-verify")
        else:
            if not args.skip_postgres:
                postgres_ok, _ = verify_postgres_counts(args)
                verified = verified and postgres_ok
            if not args.skip_redis:
                redis_ok, _ = verify_redis(args, redis_dbs)
                verified = verified and redis_ok
            if not verified:
                raise RuntimeError(
                    "Verification failed. Source cleanup is blocked. "
                    "Fix issues and rerun verification."
                )

        if args.cleanup_source_after_verify:
            if not args.skip_postgres:
                cleanup_source_postgres(args)
            if not args.skip_redis:
                cleanup_source_redis(args, redis_dbs)
            print("[cleanup] source data removed after successful verification")

        print("Migration completed.")
        return 0
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
