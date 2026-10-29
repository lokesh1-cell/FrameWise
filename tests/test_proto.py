from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from physical_ai_pipeline.frame import DropReason
from physical_ai_pipeline.output.convert import DROP_REASON_TO_PB
from physical_ai_pipeline.proto_gen import physical_ai_pb2 as pb
from physical_ai_pipeline.proto_gen.foxglove import foxglove_pb2 as fg

ROOT = Path(__file__).resolve().parents[1]


def test_compile_script_regenerates(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "compile_proto.sh")],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "USE_GRPC_TOOLS": "1", "PYTHON": sys.executable},
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert (ROOT / "src/physical_ai_pipeline/proto_gen/physical_ai_pb2.py").is_file()


def test_every_drop_reason_has_proto_value() -> None:
    assert set(DROP_REASON_TO_PB) == set(DropReason)
    assert len(set(DROP_REASON_TO_PB.values())) == len(DropReason)


def test_roundtrip_drop_event() -> None:
    msg = pb.DropEvent(
        start_frame=15, end_frame=75, reason=pb.DROP_REASON_CHILD_DETECTED, anchor_frames=[45], frame_count=61
    )
    again = pb.DropEvent.FromString(msg.SerializeToString())
    assert again == msg


def test_foxglove_schema_names() -> None:
    assert fg.CompressedImage.DESCRIPTOR.full_name == "foxglove.CompressedImage"
    assert fg.RawImage.DESCRIPTOR.full_name == "foxglove.RawImage"
    assert fg.ImageAnnotations.DESCRIPTOR.full_name == "foxglove.ImageAnnotations"
    assert {f.name for f in fg.CompressedImage.DESCRIPTOR.fields} == {"timestamp", "frame_id", "data", "format"}


def test_no_face_fields_in_schemas() -> None:
    forbidden = {"age", "embedding", "face_crop", "face_embedding"}
    for message in pb.DESCRIPTOR.message_types_by_name.values():
        assert not forbidden & {f.name for f in message.fields}, message.full_name
