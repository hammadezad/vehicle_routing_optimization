"""
Reusable decorators for timing and exception logging.
"""
import time
import logging
from functools import wraps
from typing import Optional, Type

def log_and_time(phase_name: Optional[str] = None, error_cls: Type[BaseException] = Exception, rethrow: bool = True):
    """
    Logs start/end/duration and logs exceptions with stack traces.
    On error, rethrows as error_cls by default.
    """
    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            name = phase_name or func.__name__
            t0 = time.perf_counter()
            logger.info(f"▶️ {name} start")
            try:
                result = func(*args, **kwargs)
                dt = time.perf_counter() - t0
                logger.info(f"✅ {name} done in {dt:.3f}s")
                return result
            except Exception as e:
                dt = time.perf_counter() - t0
                logger.exception(f"❌ {name} failed after {dt:.3f}s: {e}")
                if rethrow:
                    raise error_cls(str(e)) from e
                return None
        return inner
    return outer
