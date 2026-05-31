"""Reusable retry and fallback decorators powered by tenacity."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Type

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def with_exponential_retry(
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 30.0,
):
    """Class decorator: retry with exponential backoff on the given exceptions."""
    return retry(
        retry=retry_if_exception_type(exceptions),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        reraise=True,
    )


def with_fallback(fallback_fn: Callable) -> Callable:
    """Decorator: call fallback_fn with the same args if the decorated function raises."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                return fallback_fn(*args, **kwargs)

        return wrapper

    return decorator
