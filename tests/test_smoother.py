from __future__ import annotations

import random

import numpy as np
import pytest

from physical_ai_pipeline.config import FilterMode, SmoothingConfig
from physical_ai_pipeline.frame import DropReason, FrameRecord, Verdict
from physical_ai_pipeline.postprocess.smoother import TemporalSmoother

SOFT = Verdict(DropReason.BLURRY, FilterMode.SOFT, "blur")
HARD = Verdict(DropReason.PHONE_USAGE, FilterMode.HARD, "phone")
CHILD = Verdict(DropReason.CHILD_DETECTED, FilterMode.HARD, "child_detection")


def make(pattern: str) -> list[FrameRecord]:
    """'.' kept, 's' soft drop, 'h' hard drop, 'c' child drop."""
    records = []
    for i, ch in enumerate(pattern):
        r = FrameRecord(i, i, np.zeros((2, 2, 3), np.uint8))
        if ch == "s":
            r.add_verdict(SOFT)
        elif ch == "h":
            r.add_verdict(HARD)
        elif ch == "c":
            r.add_verdict(CHILD)
            r.release_pixels()
        records.append(r)
    return records


def smooth(pattern: str, gap: int = 2, island: int = 30) -> str:
    smoother = TemporalSmoother(SmoothingConfig(max_gap_fill_frames=gap, min_island_frames=island))
    out: list[FrameRecord] = []
    for r in make(pattern):
        out.extend(smoother.push(r))
    out.extend(smoother.flush())
    assert [r.index for r in out] == list(range(len(pattern)))
    return "".join("x" if r.dropped else "." for r in out)


def test_fills_short_soft_gaps() -> None:
    assert smooth("." * 40 + "ss" + "." * 40) == "." * 82
    assert smooth("." * 40 + "s" + "." * 40) == "." * 81


def test_does_not_fill_long_gaps() -> None:
    assert smooth("." * 40 + "sss" + "." * 40) == "." * 40 + "xxx" + "." * 40


def test_never_fills_child_or_hard_gaps() -> None:
    assert smooth("." * 40 + "c" + "." * 40) == "." * 40 + "x" + "." * 40
    assert smooth("." * 40 + "h" + "." * 40) == "." * 40 + "x" + "." * 40
    assert smooth("." * 40 + "sc" + "." * 40) == "." * 40 + "xx" + "." * 40


def test_edge_gaps_not_filled() -> None:
    assert smooth("s" + "." * 40) == "x" + "." * 40
    assert smooth("." * 40 + "s") == "." * 40 + "x"


def test_drops_short_islands() -> None:
    assert smooth("h" * 10 + "." * 29 + "h" * 10) == "x" * 49
    assert smooth("h" * 10 + "." * 30 + "h" * 10) == "x" * 10 + "." * 30 + "x" * 10
    assert smooth("." * 12) == "x" * 12  # whole session shorter than an island


def test_gap_fill_joins_islands() -> None:
    # 15 + 2 (filled) + 15 = 32 >= 30 -> kept as one sequence.
    assert smooth("h" * 5 + "." * 15 + "ss" + "." * 15 + "h" * 5) == "x" * 5 + "." * 32 + "x" * 5


def test_restored_frames_marked(tmp_path: object) -> None:
    smoother = TemporalSmoother(SmoothingConfig())
    records = make("." * 40 + "s" + "." * 40)
    for r in records:
        list(smoother.push(r))
    list(smoother.flush())
    assert records[40].restored and not records[40].dropped
    assert smoother.restored == 1


def reference(pattern: str, gap: int, island: int) -> str:
    dropped = [c != "." for c in pattern]
    restorable = [c == "s" for c in pattern]
    n = len(pattern)
    final = dropped[:]
    i = 0
    while i < n:
        if dropped[i]:
            j = i
            while j < n and dropped[j]:
                j += 1
            if i > 0 and j < n and j - i <= gap and all(restorable[i:j]):
                for k in range(i, j):
                    final[k] = False
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if not final[i]:
            j = i
            while j < n and not final[j]:
                j += 1
            if j - i < island:
                for k in range(i, j):
                    final[k] = True
            i = j
        else:
            i += 1
    return "".join("x" if d else "." for d in final)


@pytest.mark.parametrize("seed", range(60))
def test_streaming_matches_offline_reference(seed: int) -> None:
    rng = random.Random(seed)
    parts = []
    for _ in range(rng.randint(3, 25)):
        kind = rng.choice(".....sshc")
        length = rng.choice([1, 2, 3, 5, 12, 29, 30, 31, 45])
        parts.append(kind * length)
    pattern = "".join(parts)
    assert smooth(pattern) == reference(pattern, 2, 30)


def test_disabled_passthrough() -> None:
    smoother = TemporalSmoother(SmoothingConfig(enabled=False))
    records = make("..s..")
    out = [r for rec in records for r in smoother.push(rec)]
    assert [r.dropped for r in out] == [False, False, True, False, False]
