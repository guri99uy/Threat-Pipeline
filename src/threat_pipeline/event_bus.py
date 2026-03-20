"""In-process event bus for alert publishing.

Simple publish/subscribe pattern using callbacks.  Sufficient for the
challenge scope; production would use Redis Pub/Sub, Kafka, or NATS
for cross-process delivery to the Alarm Receiving Centre (ARC).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class EventBus:
    """Minimal in-process pub/sub event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._history: list[tuple[str, Any]] = []

    def subscribe(self, topic: str, callback: Callable) -> None:
        self._subscribers[topic].append(callback)

    def publish(self, topic: str, event: Any) -> None:
        self._history.append((topic, event))
        for cb in self._subscribers[topic]:
            cb(event)

    @property
    def history(self) -> list[tuple[str, Any]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
