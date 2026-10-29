# FrameWise 1.1 release checks

Checked on 30 October 2026, on the target Linux laptop with an Intel
i5-12450H and approximately 16 GiB RAM, using Python 3.13.

## Results

| Check | Result |
| --- | --- |
| Complete automated suite, with access to a desktop display | 186 passed; no skipped tests |
| Ruff across source and tests | Passed |
| Strict mypy across all 61 source modules | Passed |
| Installed dependency consistency (`pip check`) | Passed |
| Shell syntax: launcher, setup, protobuf compiler | Passed |
| Version 1.1 wheel build | Passed |
| Native preview → export → result → history smoke test | Passed |
| Workspace and history screenshot inspection | Passed |

## Behaviors verified

- Existing quality filters, inference postprocessing, temporal smoothing,
  cloud orchestration, protobuf messages, MCAP output, sensor export and
  provenance continue to pass their regression tests. Cloud tests use local
  fakes; real cloud resources were not deployed.
- Real H.264 input is decoded, filtered, resized and exported. The number of
  video frames agrees with the frame audit and summary report. Smaller
  inputs are not upscaled, and fractional frame rates are retained.
- Audio from a removed middle section is omitted: a three-tone input is
  checked by its resulting frequencies and output duration. Audio is mono
  AAC and follows the retained frames.
- Duplicate and static-stretch filters are checked against their expected
  boundary behavior. Disabled static filtering skips optical flow work.
- Repeated exports create distinct folders and preserve the original input.
  A fully rejected input produces reports without an invalid empty MP4.
- Cancellation during video or audio export removes partial video files.
  An injected output-write failure produces a failed status report and
  restores OpenCV thread settings.
- Audio-only inputs get a clear missing-video error. Invalid options and
  non-directory output locations do not start a UI job.
- Native interaction tests cover preview loading, valid exports, persisted
  history, empty output, stale preview events, inline validation, injected
  worker errors, retry availability, cancellation, and closing during work.
- Result actions remain visible at the minimum 980×740 window size; the
  source/settings area scrolls. Video thumbnails fit their preview viewport.
- Malformed settings and history do not prevent launch. Preferences are
  written through a temporary file and replaced atomically. Rotating logs,
  source videos, output files, virtual environments and local preferences
  are excluded from the publishing tree.

## Delivery and operation

`bash run_ui.sh` starts the app using the project virtual environment.
`bash scripts/setup_laptop.sh` prepares a new checkout. The installed
`framewise` and `pap-ui` commands are equivalent desktop entry points.
README instructions and a workspace screenshot are included.

GitHub Actions is configured to run lint, full type checking, tests with an
Xvfb display, and a wheel build on Python 3.11 and 3.13. These remote jobs
have not been claimed as passing until they actually execute on GitHub.

## Scope and limitations

This is a tested local desktop release, not a certification for every
codec, operating system or deployment environment. macOS, Windows, real
GCP infrastructure, TensorRT engines and GPU throughput were not exercised
on this Linux laptop. The local suite uses Python 3.13; other Python
versions are covered by the configured CI matrix when it runs.

Desktop mode performs video quality checks only. Semantic child, phone,
pose and activity detection require the existing model-backed pipeline.
Output uses a constant average frame rate, even pixel dimensions, the
first audio track converted to mono AAC, and no subtitle tracks. Video
rotation metadata and HDR/color-managed workflows are not specifically
validated. Variable-frame-rate inputs are normalized; severe timestamp
irregularities may need separate inspection.

Performance varies with resolution, codec, selected filters, audio and
source duration. The earlier workflow comparison is recorded in
[LAPTOP_ANALYSIS.md](LAPTOP_ANALYSIS.md); it compares different local
workflows and is not a benchmark of real AI inference.
