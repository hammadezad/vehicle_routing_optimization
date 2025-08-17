"""
Project-local logging wrapper that configures handlers and re-exports stdlib logging.
Usage in your code:
    import vrp_utils.logging as logging
    logging.setup(log_dir="logs", level="INFO", capture_print=False)
    logger = logging.getLogger(__name__)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
import logging as _stdlog
from logging.handlers import RotatingFileHandler

# Optional: include context in every log line if available
try:
    from vrp_utils.context import LogContextFilter  # same folder: vrp_utils/context.py
except Exception:
    LogContextFilter = None  # type: ignore


class _StreamToLogger(io.TextIOBase):
    """Redirect stdout/stderr to the logging system."""
    def __init__(self, logger: _stdlog.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, s):
        self._buf += str(s)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buf.strip():
            self.logger.log(self.level, self._buf.strip())
            self._buf = ""


def setup(
    log_dir: str = "logs",
    level: str = "INFO",
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
    capture_print: bool = False,
) -> _stdlog.Logger:
    """
    Configure console + rotating file handlers.
    Safe to call multiple times; guarded by a flag on the root logger.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    app_path = Path(log_dir) / "app.log"
    err_path = Path(log_dir) / "errors.log"

    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | "
        "scenario=%(scenario_id)s cluster=%(cluster_id)s subcluster=%(subcluster_id)s | "
        "%(message)s"
    )

    root = _stdlog.getLogger()
    if getattr(root, "_vrp_local_logging_initialized", False):
        return root

    root.setLevel(getattr(_stdlog, level.upper(), _stdlog.INFO))

    console = _stdlog.StreamHandler()
    console.setLevel(getattr(_stdlog, level.upper(), _stdlog.INFO))
    console.setFormatter(_stdlog.Formatter(fmt))

    file_info = RotatingFileHandler(str(app_path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_info.setLevel(getattr(_stdlog, level.upper(), _stdlog.INFO))
    file_info.setFormatter(_stdlog.Formatter(fmt))

    file_err = RotatingFileHandler(str(err_path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_err.setLevel(_stdlog.ERROR)
    file_err.setFormatter(_stdlog.Formatter(fmt))

    root.addHandler(console)
    root.addHandler(file_info)
    root.addHandler(file_err)

    if LogContextFilter is not None:
        filt = LogContextFilter()
        console.addFilter(filt)
        file_info.addFilter(filt)
        file_err.addFilter(filt)
    else:
        # Provide default values for formatter fields if context filter isn't present
        class _DefaultFieldsFilter(_stdlog.Filter):
            def filter(self, record: _stdlog.LogRecord) -> bool:
                for k, v in {"scenario_id": "-", "cluster_id": "-", "subcluster_id": "-"}.items():
                    if not hasattr(record, k):
                        setattr(record, k, v)
                return True
        df = _DefaultFieldsFilter()
        console.addFilter(df)
        file_info.addFilter(df)
        file_err.addFilter(df)

    if capture_print:
        sys.stdout = _StreamToLogger(_stdlog.getLogger("stdout"), _stdlog.INFO)
        sys.stderr = _StreamToLogger(_stdlog.getLogger("stderr"), _stdlog.ERROR)

    root._vrp_local_logging_initialized = True  # type: ignore[attr-defined]
    return root


# Re-export stdlib logging API so you can use this module like logging
getLogger = _stdlog.getLogger
DEBUG = _stdlog.DEBUG
INFO = _stdlog.INFO
WARNING = _stdlog.WARNING
ERROR = _stdlog.ERROR
CRITICAL = _stdlog.CRITICAL
exception = _stdlog.exception
