"""Custom logger with rich console formatting and rotating file output.

Every log emitted by the application — whether during normal startup, API
requests, or test runs — is also written to rotating files under
``backend/logs/``.  Files are capped at **10 MB** each and up to **5**
backups are kept (≈60 MB worst-case disk usage).
"""

from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve the *backend* package root so the logs directory lands next to src/.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # …/backend
_LOG_DIR = _BACKEND_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Rich console handler (for terminal output)
# ---------------------------------------------------------------------------
MINSY_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "debug": "dim white",
        "success": "bold green",
        "api": "bold magenta",
        "db": "bold blue",
        "agent": "bold yellow",
    }
)

console = Console(theme=MINSY_THEME)

_rich_handler = RichHandler(
    console=console,
    show_time=True,
    show_level=True,
    show_path=True,
    omit_repeated_times=False,
    log_time_format="%m/%d/%y %H:%M:%S",
    rich_tracebacks=True,
    tracebacks_show_locals=False,
    markup=True,
)
_rich_handler.setFormatter(logging.Formatter("%(message)s"))

# ---------------------------------------------------------------------------
# Rotating file handler (for persistent log files)
# ---------------------------------------------------------------------------
# Regex matching Rich markup tags used in this project.  Content brackets
# like ``[crypto]`` or ``[us_stocks]`` are intentionally NOT matched.
_RICH_TAG_RE = __import__("re").compile(
    r"\[/?"
    r"(?:success|error|api|agent|db|"
    r"bold(?:\s+\w+)?|dim|italic|underline|strike)"
    r"\]",
    __import__("re").IGNORECASE,
)


class _PlainFileFormatter(logging.Formatter):
    """Formatter that strips Rich markup tags for clean plain-text log files.

    VS Code's built-in log syntax highlighting already colours timestamps,
    level names, and bracketed logger names, so ANSI codes are not needed.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record = logging.makeLogRecord(record.__dict__)
        record.msg = _RICH_TAG_RE.sub("", record.getMessage())
        record.args = None
        return super().format(record)


_file_handler = logging.handlers.RotatingFileHandler(
    filename=_LOG_DIR / "minsy.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB per file
    backupCount=5,              # keep up to 5 rotated files
    encoding="utf-8",
)
_file_handler.setFormatter(
    _PlainFileFormatter(
        fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_file_handler.setLevel(logging.DEBUG)  # capture everything to disk


# ---------------------------------------------------------------------------
# Logger setup helpers
# ---------------------------------------------------------------------------
def setup_logger(name: str = "minsy", level: int = logging.INFO) -> logging.Logger:
    """Set up and return a configured logger instance."""
    app_logger = logging.getLogger(name)
    app_logger.setLevel(level)

    if app_logger.handlers:
        return app_logger

    app_logger.addHandler(_rich_handler)
    app_logger.addHandler(_file_handler)
    app_logger.propagate = False
    return app_logger


def configure_uvicorn_logging(level: int = logging.INFO) -> None:
    """Route uvicorn/fastapi logs through one rich handler + file handler."""
    for name in ("uvicorn", "uvicorn.error", "fastapi"):
        framework_logger = logging.getLogger(name)
        framework_logger.handlers.clear()
        framework_logger.addHandler(_rich_handler)
        framework_logger.addHandler(_file_handler)
        framework_logger.setLevel(level)
        framework_logger.propagate = False

    # Disable uvicorn access logger (handled by middleware).
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.addHandler(logging.NullHandler())
    access_logger.addHandler(_file_handler)
    access_logger.propagate = False


def configure_sqlalchemy_logging(show_sql: bool = False) -> None:
    """Keep SQLAlchemy logs styled, but hide SQL statements by default."""
    sqlalchemy_level = logging.INFO if show_sql else logging.WARNING
    for name in ("sqlalchemy", "sqlalchemy.engine", "sqlalchemy.pool"):
        sqlalchemy_logger = logging.getLogger(name)
        sqlalchemy_logger.handlers.clear()
        sqlalchemy_logger.addHandler(_rich_handler)
        sqlalchemy_logger.addHandler(_file_handler)
        sqlalchemy_logger.setLevel(sqlalchemy_level)
        sqlalchemy_logger.propagate = False


def configure_logging(level: str = "INFO", show_sql: bool = False) -> logging.Logger:
    """Configure root, app and uvicorn-family loggers."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(_rich_handler)
    root_logger.addHandler(_file_handler)
    root_logger.setLevel(log_level)

    configure_uvicorn_logging(log_level)
    configure_sqlalchemy_logging(show_sql=show_sql)

    app_logger = setup_logger(level=log_level)
    app_logger.setLevel(log_level)
    return app_logger


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
logger = setup_logger()


# ---------------------------------------------------------------------------
# Convenience helpers with rich markup
# ---------------------------------------------------------------------------
def log_api(method: str, path: str, status: int | None = None) -> None:
    """Log API request/response."""
    status_str = f" -> {status}" if status is not None else ""
    logger.info(f"[api]{method}[/api] {path}{status_str}")


def log_agent(action: str, detail: str = "") -> None:
    """Log agent activity."""
    logger.info(f"[agent]{action}[/agent] {detail}".rstrip())


def log_db(operation: str, detail: str = "") -> None:
    """Log database operations."""
    logger.debug(f"[db]{operation}[/db] {detail}".rstrip())


def log_success(message: str) -> None:
    """Log success message."""
    logger.info(f"[success]✅ {message}[/success]")


def log_error(message: str, exc: Exception | None = None) -> None:
    """Log error message."""
    logger.error(f"[error]{message}[/error]", exc_info=exc)


_PLAIN_BANNER = """\
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ███╗   ███╗██╗███╗   ██╗███████╗██╗   ██╗              ║
║   ████╗ ████║██║████╗  ██║██╔════╝╚██╗ ██╔╝              ║
║   ██╔████╔██║██║██╔██╗ ██║███████╗ ╚████╔╝               ║
║   ██║╚██╔╝██║██║██║╚██╗██║╚════██║  ╚██╔╝                ║
║   ██║ ╚═╝ ██║██║██║ ╚████║███████║   ██║                 ║
║   ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝                 ║
║                                                          ║
║   AI-Powered Quantitative Trading Platform               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝"""


def banner() -> None:
    """Print Minsy startup banner to both console and log file."""
    # ---- Rich console output ----
    console.print(
        """
[bold cyan]╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   [bold white]███╗   ███╗██╗███╗   ██╗███████╗██╗   ██╗[/bold white]              ║
║   [bold white]████╗ ████║██║████╗  ██║██╔════╝╚██╗ ██╔╝[/bold white]              ║
║   [bold white]██╔████╔██║██║██╔██╗ ██║███████╗ ╚████╔╝[/bold white]               ║
║   [bold white]██║╚██╔╝██║██║██║╚██╗██║╚════██║  ╚██╔╝[/bold white]                ║
║   [bold white]██║ ╚═╝ ██║██║██║ ╚████║███████║   ██║[/bold white]                 ║
║   [bold white]╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝[/bold white]                 ║
║                                                          ║
║   [dim]AI-Powered Quantitative Trading Platform[/dim]               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝[/bold cyan]
""",
        highlight=False,
    )
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[dim]Started at {started}[/dim]\n")

    # ---- Plain-text to log file (no formatter prefix — decorative block) ----
    _file_handler.stream.write(f"\n{_PLAIN_BANNER}\n\nStarted at {started}\n\n")
    _file_handler.stream.flush()
