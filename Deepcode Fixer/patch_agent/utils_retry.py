"""Retry utilities with exponential backoff for API calls."""
import logging
import time
from typing import Callable, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    operation_name: str = "operation",
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: Tuple of exceptions that should trigger retry
        operation_name: Name of operation for logging

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt < max_retries:
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    operation_name,
                    attempt,
                    max_retries,
                    str(exc)[:200],
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name,
                    max_retries,
                    str(exc)[:200],
                )

    raise last_exception

