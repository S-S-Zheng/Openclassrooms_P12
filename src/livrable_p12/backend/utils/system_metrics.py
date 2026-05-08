# Imports
import functools
import logging
import time
from typing import Any, Callable, Tuple

logger = logging.getLogger(__name__)


def get_duration(func: Callable) -> Callable:
    """
    Décorateur pour mesurer la latence d'une fonction.
    Retourne un tuple (résultat, durée_ms).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time = time.monotonic()
        result = func(*args, **kwargs)
        duration_ms = (time.monotonic() - start_time) * 1000
        return result, duration_ms

    return wrapper


# Version asynchrone si nécessaire pour le LLM
def get_duration_async(func: Callable) -> Callable:
    """Version asynchrone du décorateur de performance."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time = time.monotonic()
        result = await func(*args, **kwargs)
        duration_ms = (time.monotonic() - start_time) * 1000
        return result, duration_ms

    return wrapper
