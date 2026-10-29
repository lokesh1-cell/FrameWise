"""Download the InsightFace ``buffalo_l`` pack and extract the two ONNX
graphs the child detector needs: ``det_10g.onnx`` (SCRFD-10GF face detector)
and ``genderage.onnx`` (age/gender head).

    python -m physical_ai_pipeline.inference.export.fetch_insightface --out /work/onnx

Check the InsightFace model licence (non-commercial research use for the
pretrained packs) before production use.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

from physical_ai_pipeline.logging_utils import configure_logging, get_logger, log_event

_log = get_logger(__name__)

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MEMBERS = ("det_10g.onnx", "genderage.onnx")


def fetch(out_dir: Path, url: str = BUFFALO_L_URL) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https URL
        payload = resp.read()
    written: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            name = Path(info.filename).name
            if name in MEMBERS:
                target = out_dir / name
                target.write_bytes(archive.read(info))
                written.append(target)
                log_event(
                    _log,
                    "insightface_extracted",
                    file=str(target),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                )
    missing = set(MEMBERS) - {p.name for p in written}
    if missing:
        raise RuntimeError(f"buffalo_l archive missing {sorted(missing)}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default=BUFFALO_L_URL)
    args = parser.parse_args(argv)
    configure_logging()
    fetch(args.out, args.url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
