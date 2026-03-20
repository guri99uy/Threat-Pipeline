"""C#1 — Cross-turn incident state with EMA accumulation.

Tracks per-source threat score history using an Exponential Moving Average
and detects escalation via consecutive rising scores. Replaces the simple
cooldown dict in AlertDecisionEngine.
"""

from __future__ import annotations

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.models import IncidentSnapshot


class _SourceState:
    """Internal mutable state for a single source."""

    __slots__ = ("accumulated_score", "consecutive_rising", "turn_count", "prev_score")

    def __init__(self) -> None:
        self.accumulated_score: float = 0.0
        self.consecutive_rising: int = 0
        self.turn_count: int = 0
        self.prev_score: float = 0.0


class IncidentState:
    """Per-source EMA accumulation and escalation detection.

    Parameters are drawn from Settings:
        - incident_ema_alpha: EMA smoothing factor (higher = more weight on current)
        - escalation_consecutive_threshold: consecutive rising turns to flag escalation
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._sources: dict[str, _SourceState] = {}

    def update(self, source_id: str, current_score: float) -> IncidentSnapshot:
        """Update state for *source_id* with the latest *current_score*.

        Returns an immutable snapshot of the new state.
        """
        alpha = self.settings.incident_ema_alpha

        if source_id not in self._sources:
            self._sources[source_id] = _SourceState()

        s = self._sources[source_id]
        s.turn_count += 1

        if s.turn_count == 1:
            # First observation — seed the EMA
            s.accumulated_score = current_score
        else:
            s.accumulated_score = alpha * current_score + (1 - alpha) * s.accumulated_score

        # Consecutive rising detection
        if current_score > s.prev_score and s.turn_count > 1:
            s.consecutive_rising += 1
        else:
            s.consecutive_rising = 0

        s.prev_score = current_score

        is_escalating = s.consecutive_rising >= self.settings.escalation_consecutive_threshold

        return IncidentSnapshot(
            source_id=source_id,
            accumulated_score=s.accumulated_score,
            consecutive_rising=s.consecutive_rising,
            turn_count=s.turn_count,
            is_escalating=is_escalating,
        )

    def get_state(self, source_id: str) -> IncidentSnapshot | None:
        """Return the current snapshot for *source_id*, or None if unseen."""
        s = self._sources.get(source_id)
        if s is None:
            return None
        return IncidentSnapshot(
            source_id=source_id,
            accumulated_score=s.accumulated_score,
            consecutive_rising=s.consecutive_rising,
            turn_count=s.turn_count,
            is_escalating=s.consecutive_rising >= self.settings.escalation_consecutive_threshold,
        )
