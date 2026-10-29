#!/usr/bin/env python3
"""Verify that session .mcap files load the way Foxglove Studio reads them.

    python tools/verify_mcap.py out/session.mcap [more.mcap ...]

Exit code 0 when every file passes. To inspect visually: open the file in
Foxglove Studio (File > Open local file), add an Image panel on
/camera/image_compressed with /camera/annotations enabled, a Raw Messages
panel on /detections and a Plot panel on /quality.laplacian_variance.
"""

from __future__ import annotations

import sys

from physical_ai_pipeline.verify import verify_mcap


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    ok = True
    for path in argv:
        report = verify_mcap(path)
        print(report.to_json())
        ok &= report.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
