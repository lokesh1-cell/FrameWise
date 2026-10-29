"""Tier 3 decision filters operating on cached Tier 2 inference."""

from __future__ import annotations

from physical_ai_pipeline.config import Tier3Config
from physical_ai_pipeline.filters.base import FilterChain, FilterEventListener
from physical_ai_pipeline.filters.tier3.child import (
    ChildDetectionFilter,
    ChildEvent,
    exclusion_range,
    merge_ranges,
)
from physical_ai_pipeline.filters.tier3.fake_work import FakeWorkFilter
from physical_ai_pipeline.filters.tier3.off_task import OffTaskFilter
from physical_ai_pipeline.filters.tier3.phone import PhoneUsageFilter

__all__ = [
    "ChildDetectionFilter",
    "ChildEvent",
    "FakeWorkFilter",
    "OffTaskFilter",
    "PhoneUsageFilter",
    "build_decision_chain",
    "exclusion_range",
    "merge_ranges",
]


def build_decision_chain(
    config: Tier3Config, listener: FilterEventListener | None, task_whitelist: tuple[str, ...] | None
) -> FilterChain:
    """Phone -> off-task -> fake work. The child filter runs separately,
    before this chain, because it needs the transient face results."""
    return FilterChain(
        [
            PhoneUsageFilter(config.phone, listener),
            OffTaskFilter(config.off_task, listener, task_whitelist),
            FakeWorkFilter(config.fake_work, listener),
        ]
    )
