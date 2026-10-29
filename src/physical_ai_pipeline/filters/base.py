"""Filter abstraction with SHADOW / SOFT / HARD modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from physical_ai_pipeline.config import FilterMode, FilterSettings
from physical_ai_pipeline.frame import DropReason, FrameRecord, Verdict
from physical_ai_pipeline.logging_utils import get_logger
from physical_ai_pipeline.timeline import FrameWindow

_log = get_logger(__name__)

SettingsT = TypeVar("SettingsT", bound=FilterSettings)


@dataclass(frozen=True, slots=True)
class FilterHit:
    """A filter decision covering frames [start, end] (inclusive)."""

    reason: DropReason
    start: int
    end: int
    detail: Mapping[str, float] = field(default_factory=dict)
    anchor_frame: int | None = None


@dataclass(frozen=True, slots=True)
class FilterEvent:
    filter_name: str
    mode: FilterMode
    hit: FilterHit
    pts_ns: int


class FilterEventListener(Protocol):
    def on_filter_event(self, event: FilterEvent) -> None: ...


class FilterBase(ABC, Generic[SettingsT]):
    """Base class for every Tier 1 and Tier 3 filter.

    Subclasses implement :meth:`evaluate`, returning hits. :meth:`process`
    turns hits into verdicts according to the configured mode:

    * SHADOW: verdict stored in ``record.shadow_verdicts`` only (frame kept).
    * SOFT:   frame dropped, restorable by the temporal smoother.
    * HARD:   frame dropped permanently.
    """

    name: ClassVar[str]
    # Free pixel buffers of affected frames as soon as a hit is applied.
    release_pixels_on_hit: ClassVar[bool] = False

    def __init__(self, settings: SettingsT, listener: FilterEventListener | None = None) -> None:
        self.settings = settings
        self._listener = listener
        self.hits = 0

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def mode(self) -> FilterMode:
        return self.settings.mode

    @abstractmethod
    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        """Inspect ``record`` (and internal state) and return any hits."""

    def process(self, record: FrameRecord, window: FrameWindow) -> bool:
        """Evaluate and apply. Returns True when ``record`` is now dropped."""
        if not self.enabled:
            return False
        try:
            hits = self.evaluate(record)
        except Exception:
            # A broken filter must not kill the session; the frame continues
            # through the remaining filters and the error is logged.
            _log.exception(
                "filter_error",
                extra={"fields": {"event": "filter_error", "filter": self.name, "frame_index": record.index}},
            )
            return record.dropped
        for hit in hits:
            self._apply(hit, record, window)
        return record.dropped

    def _apply(self, hit: FilterHit, record: FrameRecord, window: FrameWindow) -> None:
        self.hits += 1
        verdict = Verdict(
            reason=hit.reason,
            mode=self.mode,
            filter_name=self.name,
            detail=tuple(sorted((k, float(v)) for k, v in hit.detail.items())),
        )
        release = self.release_pixels_on_hit and self.mode is not FilterMode.SHADOW
        if hit.start == hit.end == record.index:
            record.add_verdict(verdict)
            if release:
                record.release_pixels()
        else:
            window.apply_range(hit.start, hit.end, verdict, release_pixels=release)
            if hit.start <= record.index <= hit.end and window.get(record.index) is not record:
                record.add_verdict(verdict)
                if release:
                    record.release_pixels()
        if self._listener is not None:
            self._listener.on_filter_event(FilterEvent(self.name, self.mode, hit, record.pts_ns))


class FilterChain:
    """Runs filters in order and stops at the first non-shadow drop (fail fast)."""

    def __init__(self, filters: Sequence[FilterBase[Any]]) -> None:
        self.filters = list(filters)

    def process(self, record: FrameRecord, window: FrameWindow) -> bool:
        """Returns True when the frame survives the chain."""
        for flt in self.filters:
            if flt.process(record, window):
                return False
        return True
