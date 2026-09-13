# VTS Merserk Temporal Enhance

Enhances an ordered sequence of losslessly compressed images while retaining neural
history between frames. There are no video-file, frame-rate, audio, HDR or outer
iteration settings. Frame count and ordering are unchanged.

## Processing

VTS uses the shared Scale To Min sizing/crop rules, including reversed side limits,
divisibility and local Lanczos downscaling. If one axis shrinks and the other grows,
only the shrinking axis is reduced locally. The server opens one RTX VSR session
if enlargement is needed, then one Neuroframe session at the final dimensions.

All neural frames use that same session and consecutive indices. Upstream's scene
cut detector runs on source frames before VSR/neural changes. History resets on the
first frame and detected cuts, not on each uploaded image or network chunk.

`nr_passes` controls native passes per frame, from 1 to 4. `shimmer_suppression`
controls upstream temporal stabilization from 0 to 1. This stream uses the upstream
RGBA frame boundary: the native bridge estimates motion, and the host path uses
DIS optical flow to stabilize the final composed enhancement. It does not use the
GPU video decode/encode path, because its inputs and outputs are PNG images.

The usual style, intensity, tone/structure and preservation controls remain
available. Skin Structure requires Automatic Mask; no automatic widget toggle is
added. The initial interface does not accept custom masks or alternate optical-flow
models. Stabilization reduces inconsistent detail but cannot guarantee artifact-free
results on every motion or scene cut.

## Transport

WebSocket route: `/vts/enhance_sequence`. Use ws/wss corresponding to the server's
http/https URL. Version 1 negotiates:

- `width`, `height`: dimensions after client cropping/local downscaling.
- `target_width`, `target_height`: exact final dimensions.
- `channels`: 3 (RGB) or 4 (RGBA); `frame_count`: positive uint32.
- `enable_neural_rendering`, `vsr_quality` (1-4), `timeout_seconds` (1-86400).
- `parameters`: supported neural controls including `nr_passes` and
  `shimmer_suppression`; no `iterations` or HDR fields.

Server replies `ready` with output dimensions, channels, frame count and a 256 KiB
chunk limit. Each upload has a JSON `frame` header with sequential `index` and PNG
`bytes`, followed by binary chunks. The final upload is followed by `{ "type": "end" }`.
For each input, the server replies with an `enhanced` header (`index`, `bytes`,
`scene_cut`) and PNG chunks. `done` follows successful native cleanup and includes
aggregate frame, cut, VSR and neural evaluation statistics. Errors send `error` and
close the stream. Each new request starts fresh history.

Fatal neural worker failures include `code: NEURAL_WORKER_RESTARTED`. The updated
VTS node replays the entire sequence once on a fresh connection, rebuilding VSR
and neural history from original inputs. See [worker recovery](WORKER_RECOVERY.md).

The client allows at most two outstanding uploads. The server has one queued frame
and processes GPU work on one dedicated worker thread. GPU calls are serialized
with existing Merserk jobs; disconnects cancel their own request and release native
resources before the next request can claim the GPU slot. GUI and API jobs wait in
the same FIFO queue. A setup with `queue_status: true` receives periodic `queued`
messages with a one-based `position` before `ready`; updated VTS clients extend
their readiness deadline on each message. Legacy clients wait silently up to their
readiness timeout. Queued disconnects remove only that request.

No server image, video, manifest or ZIP output files are created. PNG encoding and
stage handoff happen in memory; diagnostic logs may still be written. PNG preserves
8-bit channel values losslessly. ComfyUI float input is quantized to 8-bit for the
runtime; this is not a float/HDR preservation path. RGBA alpha is retained and resized
with Lanczos when necessary.

## VTS node

Find **VTS Merserk Temporal Enhance** under **VTS/video**. It accepts an IMAGE batch
or VTS DiskImage sequence. `return_type=Input` preserves the storage type; Tensor
holds the complete output in RAM, while DiskImage writes final PNG frames
incrementally on the ComfyUI client. Pure reduction/bypass works without the server
when neural rendering is off. A single frame can still be enhanced.

New defaults are native NR Passes 1, Shimmer Suppression 0.7, Scale to Min with
512/512 limits, scaling and enhancement enabled. Every option has a tooltip.

## Checks

- `python -m unittest discover -s tests -p "test_*.py"`: includes session/history,
  scene resets, chunk protocol, GPU thread affinity, disconnects and busy isolation.
- `python tests/check_temporal_gpu.py`: real neural/VSR sessions, temporal execution,
  scene cut resets, native pass counts, alpha, and memory-only image output.
- VTS `tests/test_merserk_temporal.py`: node transport, storage, resizing, failure
  cleanup and input interface without starting the ComfyUI application.
