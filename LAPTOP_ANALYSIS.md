# Laptop analysis and validation

The inspected project is `cloud-qa-automation-main`. The host reports an
Intel Core i5-12450H (12 logical CPUs) and approximately 16 GiB RAM. No
`nvidia-smi` executable was available, so CUDA/TensorRT acceleration was
not assumed. The system Python lacks both Tkinter and the video dependencies;
the existing Anaconda Python includes Tkinter. A local `.venv` was created
from that interpreter and the project dependencies were installed there.

## Findings

The original pipeline is designed for GCP workers and NVIDIA TensorRT
engines. Its default local backend supplies scripted, simulated inference
results. Real model weights and TensorRT engines are not included. CPU
quality metrics are already reusable, but there was no interactive UI.

The full path retains full-resolution frames for batching, retroactive
decisions, decoding prefetch, and smoothing. Its bounded window can still
consume substantial laptop memory: one 1080p BGR frame is about 5.9 MiB.
MP4 output reads JPEG images back from a completed MCAP and encodes them a
second time. This is useful for the cloud dataset workflow, but adds work
when the desired local artifact is a cleaned video.

## Changes

The separate laptop path reuses the Tier 1 filter implementations without
pretending that simulated detections constitute AI inference. It decodes
with two threads, resizes before allocating BGR arrays, processes frames
sequentially, and writes H.264 directly using two encoder threads and the
`veryfast` preset. Default analysis width remains 640 to retain the existing
blur calibration. Static detection is optional and optical flow is now
skipped when that filter is disabled. Full inference and its configuration
constraints remain available through the original pipeline.

Audio processing streams decoded samples and reads retained frame decisions
from disk, keeping PCM memory bounded instead of loading an entire audio
track. It converts the first track to mono 48 kHz AAC. Frame audits stream
to JSONL; summary drop counts stay in memory. Each run uses a unique folder,
so repeated inputs do not overwrite prior output. Cancellation and failures
remove incomplete video files and write a status report.

The decoder's existing prefetch helper also stops its producer and closes
its source when the consumer exits early, fixing a producer that could
remain blocked trying to enqueue its sentinel into a full queue.

## Measurements on this laptop

A generated 120-frame, 1920×1080, 30 fps H.264 clip without audio was used.
Wall time includes process startup and export; memory is peak resident set
size from `/usr/bin/time`. These are single-run measurements, not guarantees
for arbitrary videos.

| Mode | Wall time | Peak memory | Kept frames |
| --- | ---: | ---: | ---: |
| Laptop default, width capped at 1280 | 2.38 s | 216 MiB | 115 |
| Original `pap run`, default fake backend | 5.36 s | 953 MiB | 95 |

For this sample the laptop workflow took about 56% less wall time and 77%
less peak memory. The workflows perform different work: the original creates
MCAP and provenance, uses simulated inference, and applies smoothing that
removes additional frames; the laptop workflow produces a quality-only MP4
at reduced resolution. This is a comparison of local workflows, not an AI
model throughput comparison. Audio and original-resolution output add cost.

## Validation

- 159 tests passed, including the existing pipeline suite and new real-video
  checks for frame audit/output agreement, resizing, cancellation cleanup,
  all-filtered output, unique folders, audio duration, disabled optical flow,
  and prefetch shutdown.
- Ruff passed across source and tests. Mypy passed for the new processing
  and desktop modules and the modified decoder.
- A native desktop smoke test filled the input and output fields, invoked
  processing, and verified the saved result state. The 640×360 sample kept
  115 of 120 frames. The UI was captured and visually inspected.

No real TensorRT inference or GPU performance was validated because the
required hardware/runtime and model engines were not established.
