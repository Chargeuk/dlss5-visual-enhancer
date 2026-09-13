# VTS integration

This fork uses the current upstream Neuroframe, RTX VSR and DLSS Frame Generation
engines. The VTS layer supplies image APIs, interpolation streaming, enhancement
iterations and optional no-save output. No v7 compatibility runtime is required.

## Image processing

The VTS node calculates dimensions with its existing shared Scale To Min helper,
including reversed limits, divisibility and optional centre cropping. It performs
Lanczos reduction locally, including shrinking only the necessary axis for a
mixed-aspect resize. This keeps uploads small.

The server receives one PNG per source image. It uses RTX VSR once if enlargement
is needed, then feeds that result through Neuroframe enhancement iterations at the
final dimensions. Native `nr_passes` (1-4) applies within each iteration:
`iterations=3, nr_passes=2` means six neural evaluations. There is one final PNG
response, with no intermediate image files or repeated network transfers.

Both scaling and neural enhancement can be independently disabled by the VTS node.
Scaling-only requests use RTX VSR for enlargement; purely local reductions and
bypass need no server. The input/output socket accepts IMAGE or VTS DiskImage;
DiskImage writes only final results on the ComfyUI client.

## API

`/vts_enhance_memory`, `/vts_enhance` and `/vts_cancel` retain their transport
signatures and request-specific cancellation. The file endpoint uses Gradio's
cache; the memory endpoint accepts and returns base64 PNG directly without image
cache or output files. Runtime diagnostic logging remains enabled.

Image JSON accepts `operation` (`neural` or `vsr`), `target_width`, `target_height`
and `vsr_quality` (1-4). The neural operation also accepts `iterations`, `nr_passes`,
`nr_style`, `nr_intensity`, `local_tone_strength`, `local_structure_strength`,
`skin_structure_strength`, `automatic_mask`, `nr_color_strength`,
`tone_preservation`, `face_skin_protection`, `grain_preservation` and `nr_gpu_mode`.
Old DLSS presets, scaling-performance presets and renderer-selection fields are
not part of this interface. Custom-mask upload is outside this initial API.

Neural targets must be at least 64 pixels per side, at most 7680 on the longest
side and 4320 on the shortest. RTX VSR is limited to 16384 pixels per side; GPU
memory may impose lower practical limits. PNG transport preserves 8-bit pixels,
not arbitrary floating-point/HDR values.

`/vts/interpolate` remains unchanged: lossless chunked PNG, bounded buffers,
persistent GPU sessions and generated-frame-only responses. See
[VTS interpolation](VTS_INTERPOLATION.md).

## GUI and upstream code

Upstream's rendering engines, native NR Passes, composition controls, temporal
video/Live processing, GPU/RAM paths and overlapped batch processing are retained.
Our image iteration control and PNG/default versus Do not save choices remain.
GUI previews can use Gradio's cache; VTS memory API responses bypass it.

## Setup and verification

Copy `bin/` from the upstream v8.0 portable release into the source checkout,
then install `requirements-vts.txt` with its portable Python. Start with
`start.bat`. Port defaults to 7865; set `GRADIO_SERVER_NAME` to the server's LAN
address for network use. Runtime binaries, local settings and outputs are not
committed. Only the current upstream runtime is needed.

Checks (use `bin/python-3.13.15-embed-amd64/python.exe`):

- `-m unittest discover -s tests -p "test_*.py"`: stage ordering, both loop settings,
  mixed resizing, cancellation, no-save, settings and interpolation streaming.
- `tests/check_image_backend_gpu.py`: real VSR then Neuroframe, both loops,
  preservation controls, alpha, mixed resizing, GUI PNG/no-save behaviour.
- `tests/check_public_api.py http://192.168.1.1:7865`: deployed image APIs and
  interpolation stream.
- `tests/check_interpolation_gpu.py`: native interpolation multipliers.
