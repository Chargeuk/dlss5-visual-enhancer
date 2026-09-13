# VTS integration and upstream merge

`main` includes upstream commit `6000792913460fffc002731b5b750a1d14114859`
(v8-era Neuroframe backend) plus the Chargeuk VTS integrations. The tested v7
installation is preserved separately on branch `vts-merserk-v7`.

## Features retained

- Upstream Neuroframe rendering, native NR Passes, composition/mask controls,
  GPU/RAM paths, temporal stabilization, Live/video features, and overlapped
  image decoding, rendering and output handling remain intact.
- Image enhancement iterations feed each result into the next iteration. Resize
  happens once; only the final image is saved. Iterations persist through settings
  and presets. Each iteration uses the selected upstream NR Passes (1-4), so the
  total neural evaluations are `iterations * nr_passes`.
- Neural and RTX VSR image GUIs offer **Do not save**, while defaulting to PNG.
  No-save returns previews without publishing production images/manifests/ZIPs.
  Browser previews may still use Gradio's own cache. VTS memory APIs bypass that
  image cache and return lossless PNG directly.
- `/vts_enhance`, `/vts_enhance_memory`, `/vts_cancel` retain their existing
  arguments and return types. Cancellation targets the identified request.
- `/vts/interpolate` retains lossless chunked PNG transport, persistent GPU
  sessions, generated-frame-only responses, scene-cut handling and bounded
  buffers. Cleanup is shielded from ASGI cancellation before releasing the GPU.
  See [interpolation protocol and node usage](VTS_INTERPOLATION.md).

## Renderer selection and compatibility

Image API JSON accepts optional `backend`: `auto` (default), `neuroframe`, `legacy`.

- **Auto:** uses Neuroframe for compatible native-size/downscale enhancement.
  Requests for upscaling, non-default `nr_preset`, or non-default
  `dlss_model_preset` use the isolated v7 worker, preserving the existing VTS
  node's DLSS scaling and preset semantics. Explicit larger target dimensions
  also select this compatibility path. There is no silent substitution of
  Lanczos or RTX VSR for an old DLSS SR request.
- **Neuroframe:** exposes new `nr_passes`, `nr_color_strength`,
  `tone_preservation`, `face_skin_protection`, `grain_preservation`,
  `mask_feather`, and `nr_gpu_mode` in addition to existing compatible controls.
  Legacy-only scaling/preset requests produce a clear error in this mode.
- **Legacy:** uses `src/legacy/`, isolated from upstream renderer code and runtime
  directories. New Neuroframe-only controls are rejected, not silently ignored.
- **`operation=vsr`:** continues to use the upstream RTX Video Super Resolution
  image implementation with explicit target dimensions and no output-file save.

The compatibility backend is retained source from upstream v7 plus our tested
customizations, with adapters for the current decoder and shared job management.
It exists because upstream removed DLSS SR and the older preset controls from
its replacement neural renderer. Selecting a different backend can change image
appearance; the APIs preserve settings rather than promising identical output
across different renderers.

## Running a source checkout

1. Obtain the [upstream v8.0 portable release](https://github.com/Merserk/dlss5-visual-enhancer/releases/tag/v8.0)
   and copy its `bin/` directory into this checkout. Keep the source from this
   branch; do not overwrite it with release source.
2. Install the streaming dependency using the portable Python:

   ```powershell
   .\bin\python-3.13.15-embed-amd64\python.exe -m pip install -r requirements-vts.txt
   ```

3. To retain all older VTS DLSS scaling/preset workflows, obtain the
   [v7.0 release archive](https://github.com/Merserk/dlss5-visual-enhancer/releases/tag/v7.0)
   and install its compatibility runtime into a separate folder:

   ```powershell
   .\bin\python-3.13.15-embed-amd64\python.exe tools\install_legacy_runtime.py "path\to\DLSS.5.Visual.Enhancer.v7.0.zip"
   ```

4. Start with `start.bat`. Port defaults to 7865. The bind address defaults to
   loopback. For LAN use, set `GRADIO_SERVER_NAME` to an address belonging to the
   server before starting; the existing installation uses `192.168.1.1`.
   `GRADIO_SERVER_PORT` can override the port. Use an appropriate LAN firewall
   rule. Runtime binaries, configuration, outputs, caches and logs are ignored
   by Git and are not included in source commits.

The existing VTS nodes require no changes for their current image and interpolation
requests. The new Neuroframe-only controls can be supplied by API callers; adding
corresponding VTS GUI widgets is separate work.

## Verification

Use the portable Python above (shown as `python` below):

- `python -m unittest discover -s tests -p "test_*.py"`: routing, settings/presets,
  iterative feedback, no-save output, cancellation, streaming order/chunks,
  concurrent-request isolation and disconnect cleanup.
- `python tests/check_image_backend_gpu.py`: real Neuroframe, legacy preset and
  scaling, VSR, iterations, alpha preservation, GUI no-save and final PNG output.
- `python tests/check_interpolation_gpu.py`: real 2x, approximate 3x, 4x and 8x.
- `python tests/check_public_api.py http://127.0.0.1:7866`: live Gradio image,
  cancellation and WebSocket interpolation contracts against a running server.

GPU checks need the matching runtime and an idle GPU service. They may write
runtime diagnostic logs. Explicit saved-image checks use temporary directories.
