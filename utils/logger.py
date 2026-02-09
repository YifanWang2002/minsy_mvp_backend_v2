"""Custom logger with rich formatting for development debugging."""

import logging
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for Minsy
MINSY_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "debug": "dim white",
    "success": "bold green",
    "api": "bold magenta",
    "db": "bold blue",
    "agent": "bold yellow",
})

console = Console(theme=MINSY_THEME)

# Shared RichHandler for all loggers
_rich_handler = RichHandler(
    console=console,
    show_time=True,
    show_path=True,
    rich_tracebacks=True,
    tracebacks_show_locals=True,
    markup=True,
)
_rich_handler.setFormatter(logging.Formatter("%(message)s"))


def setup_logger(name: str = "minsy", level: int = logging.DEBUG) -> logging.Logger:
    """Setup and return a configured logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.addHandler(_rich_handler)
    logger.propagate = False

    return logger


def configure_uvicorn_logging() -> None:
    """Configure uvicorn and fastapi loggers to use our rich handler."""
    # Loggers to use rich handler
    rich_loggers = [
        "uvicorn",
        "uvicorn.error",
        "fastapi",
    ]

    for name in rich_loggers:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(_rich_handler)
        uvicorn_logger.propagate = False

    # Disable uvicorn access logger (we handle it in middleware)
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.addHandler(logging.NullHandler())
    access_logger.propagate = False


# Global logger instance
logger = setup_logger()

# Configure uvicorn logging on import
configure_uvicorn_logging()


# Convenience functions with rich styling
def log_api(method: str, path: str, status: int | None = None) -> None:
    """Log API request/response."""
    status_str = f" → {status}" if status else ""
    logger.info(f"[api]{method}[/api] {path}{status_str}")


def log_agent(action: str, detail: str = "") -> None:
    """Log agent activity."""
    logger.info(f"[agent]🤖 {action}[/agent] {detail}")


def log_db(operation: str, detail: str = "") -> None:
    """Log database operations."""
    logger.debug(f"[db]📦 {operation}[/db] {detail}")


def log_success(message: str) -> None:
    """Log success message."""
    logger.info(f"[success]✅ {message}[/success]")


def log_error(message: str, exc: Exception | None = None) -> None:
    """Log error message."""
    logger.error(f"[error]{message}[/error]", exc_info=exc)


def banner() -> None:
    """Print Minsy startup banner."""
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
    console.print(f"[dim]Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
