from __future__ import annotations

import numpy as np

from physical_ai_pipeline.config import GraspConfig
from physical_ai_pipeline.frame import Detection, FrameInference, FrameRecord, Hand, HandDetection
from physical_ai_pipeline.postprocess.grasp import GraspScorer, InteractionState


def frame(index: int, hand_x: float) -> FrameRecord:
    r = FrameRecord(index, index * 33_000_000, None)
    r.inference = FrameInference(
        objects=[Detection((100, 100, 140, 140), 0.9, 41, "cup")],
        hands=[HandDetection((hand_x, 100, hand_x + 40, 140), 0.9, np.zeros((21, 3), np.float32), Hand.LEFT)],
    )
    return r


def test_state_sequence_and_segment() -> None:
    scorer = GraspScorer(GraspConfig())
    xs = [20, 35, 50, 65, 80] + [100] * 12 + [150, 160, 170, 260, 300, 340]
    states = [scorer.score(frame(i, x))[0].state for i, x in enumerate(xs)]
    assert states[0] == InteractionState.IDLE
    assert InteractionState.APPROACHING in states[1:5]
    assert states[6] == InteractionState.GRASPING
    assert InteractionState.HOLDING in states
    assert InteractionState.RELEASING in states
    segments = scorer.finish()
    assert len(segments) == 1
    seg = segments[0]
    assert seg.object_class == "cup" and seg.hand is Hand.LEFT
    assert seg.phase_counts[InteractionState.HOLDING] > 0
    assert 0 < seg.mean_grasp_confidence <= 1


def test_no_contact_no_segment() -> None:
    scorer = GraspScorer(GraspConfig())
    for i in range(30):
        scorer.score(frame(i, 400))
    assert scorer.finish() == []


def test_frame_gap_closes_segment() -> None:
    scorer = GraspScorer(GraspConfig())
    for i in range(10):
        scorer.score(frame(i, 100))
    for i in range(50, 60):
        scorer.score(frame(i, 100))
    assert len(scorer.finish()) == 2
