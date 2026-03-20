"""Abstract engine base with automatic latency measurement.

Every engine inherits Engine[TIn, TOut] and implements `process()`.
Callers use `engine.run(input)` which wraps `process()` with timing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class Engine(ABC, Generic[TIn, TOut]):
    """Base class for all pipeline engines."""

    name: str = "unnamed"

    def __init__(self) -> None:
        self.last_latency_s: float = 0.0

    @abstractmethod
    def process(self, input_data: TIn) -> TOut:
        """Implement the engine's core logic."""
        ...

    def run(self, input_data: TIn) -> TOut:
        """Execute process() with latency measurement."""
        start = time.perf_counter()
        result = self.process(input_data)
        self.last_latency_s = time.perf_counter() - start
        return result
