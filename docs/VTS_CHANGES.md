# VTS integration

This branch snapshots the working v7.0 installation before migration to the newer
upstream native backend. The original release's 86 source/startup/license files
match upstream f9064948cca31aaf7e2833f468a1c76fc7141621 after line-ending normalization.

Included customizations:

- Image iterations: upscale on the first pass, enhance at the same size thereafter,
  and save only the final image. Iterations persist through settings and presets.
- Explicit target dimensions for the neural image API.
- VTS image endpoints: `/vts_enhance`, `/vts_enhance_memory`, and `/vts_cancel`.
  Memory transfers use base64 PNG and avoid the Gradio file cache.
- Neural and RTX VSR image APIs return images without saving server output files.
  Both image GUIs offer `Do not save` and default to PNG.
- Lossless PNG WebSocket interpolation at `/vts/interpolate`, persistent GPU
  sessions, generated-frame-only responses, cancellation and scene-cut handling.
  See [protocol and node usage](VTS_INTERPOLATION.md).
- LAN serving at 192.168.1.1:7865 in the captured installation. The launch bind
  address is in app.py and the port is selected in start.bat. Other installations
  must choose an address belonging to their server and an appropriate firewall rule.

The runtime binaries are supplied by the matching upstream release, not Git.
Install `requirements-vts.txt` into that runtime for WebSocket support.

Tests use the application Python environment:

    python -m unittest discover -s tests -p "test_*.py"

`tests/check_interpolation_gpu.py` and `tests/check_image_backend_gpu.py` are
explicit GPU integration checks and require the matching native runtime in bin/.
The GPU tests can write diagnostic logs; the image check also validates explicit
PNG output inside a temporary directory.
