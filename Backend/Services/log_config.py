import sys
import logging
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "Frontend" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# ── Formatters ─────────────────────────────────────────────────
FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Module-level singletons ─────────────────────────────────────
_file_handler: logging.FileHandler | None = None
_console_handler: logging.StreamHandler | None = None
_initialized: bool = False


class _ImmediateFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class _ImmediateStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class _PrintCapture:
    """
    Replaces sys.stdout / sys.stderr so that print() calls
    are captured into the log file in addition to the terminal.
    """

    def __init__(self, original_stream, log_name: str, level: int):
        self.original = original_stream
        self._logger = logging.getLogger(log_name)
        self._level = level
        self._buf = ""

    def write(self, text: str):
        # Always write to real terminal immediately
        if self.original:
            self.original.write(text)
            self.original.flush()

        # Accumulate into buffer; emit complete lines
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            clean = line.rstrip("\r")
            if clean.strip():
                self._logger.log(self._level, clean)

    def flush(self):
        if self._buf.strip():
            self._logger.log(self._level, self._buf.strip())
            self._buf = ""
        if self.original:
            self.original.flush()

    def isatty(self):
        return hasattr(self.original, "isatty") and self.original.isatty()

    # Make it behave like a real file for libraries that check
    def fileno(self):
        return self.original.fileno() if self.original else -1


def init(level: int = logging.DEBUG) -> None:
    """
    Call ONCE at process start (top of server.py, before any other import).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _file_handler, _console_handler, _initialized

    if _initialized:
        return

    # ── Handlers ────────────────────────────────────────────────
    _file_handler = _ImmediateFileHandler(LOG_FILE, mode="w", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))

    _console_handler = _ImmediateStreamHandler(sys.__stdout__)
    _console_handler.setLevel(logging.DEBUG)
    _console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))

    # ── Root logger ─────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(_file_handler)
    root.addHandler(_console_handler)

    # ── Make sure ALL child loggers propagate to root ────────────
    # (don't set propagate=False anywhere — just let root handle it)
    for name in list(logging.Logger.manager.loggerDict):
        child = logging.getLogger(name)
        child.handlers.clear()  # remove any stale handlers
        child.propagate = True  # always bubble up to root
        child.setLevel(logging.DEBUG)

    # ── Capture print() → log ───────────────────────────────────
    sys.stdout = _PrintCapture(sys.__stdout__, "STDOUT", logging.INFO)
    sys.stderr = _PrintCapture(sys.__stderr__, "STDERR", logging.ERROR)

    _initialized = True

    # ── Banner ──────────────────────────────────────────────────
    root.info("=" * 60)
    root.info("LOGGING INITIALIZED")
    root.info(f"LOG FILE : {LOG_FILE.absolute()}")
    root.info(f"LEVEL    : {logging.getLevelName(level)}")
    root.info("=" * 60)
    _file_handler.flush()


def flush() -> None:
    """Force-flush both handlers. Call after critical log lines."""
    if _file_handler:
        _file_handler.flush()
    if _console_handler:
        _console_handler.flush()
