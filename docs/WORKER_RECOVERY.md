# Neural worker recovery

Merserk's web process retains the GUI, request queue and HTTP/WebSocket services.
Neural DLLs and CUDA/D3D12 sessions run in child processes. Host-frame image jobs
use one reusable worker; file-video jobs execute their complete decode/render/
encode pipeline there so CUDA AVFrames never cross process boundaries. Live uses
its own worker, allowing playback to remain open after releasing the GPU queue.

The supervisor detects process exit, fatal native state and an unresponsive worker.
It terminates that worker and its descendants using a Windows Job Object. Future
jobs initialize a fresh worker; the Python web server remains running. Cancellation
also terminates the affected worker when native work is in progress. It does not
retry cancelled jobs or ordinary input/validation errors.

- Independent image frames (`reset=True`) get at most one automatic recovery per
  logical neural session. The same input to the failed evaluation is retried;
  completed outer iterations are not needlessly repeated.
- VTS temporal streams return `code: NEURAL_WORKER_RESTARTED` on worker loss. The
  updated node reconnects and replays the complete sequence once, using its original
  Tensor/DiskImage inputs. The new request joins the normal queue. Partial client
  output is removed before replay; only a complete attempt is returned. Both VSR
  and neural history are rebuilt. No unbounded server replay cache is introduced.
- File video retries once from the source file. Each attempt writes into a private
  directory; only a completed output is published without replacing existing files.
- Live retries once by reopening its source. A live source cannot guarantee replay
  of frames already broadcast, and playback may have a discontinuity on recovery.

Host-frame transfer uses two fixed shared-memory RGBA8 buffers per session. It
adds local memory copies, not network traffic or PNG encode/decode work. Network
transport remains lossless PNG. No server image output or disk replay cache is
created for VTS calls. Worker-owned temporary files used by video pipelines are
cleaned after worker termination; NVIDIA may retain locked extracted DLL cache
files, so cache cleanup is best effort and cannot block recovery. Windows Job Objects also kill owned decoder/
encoder processes if the supervising web process exits.

This contains a failed native runtime; it does not repair a persistent GPU/driver
fault. If a fresh worker cannot render, the bounded retry fails visibly and the
next queued request can proceed. Requests that already returned an error before
this implementation still require resubmission. The new VTS node must be loaded
by restarting ComfyUI to enable automatic temporal replay.

Validation includes unit tests for queue/retry limits, cancellation and output
cleanup; real GPU worker termination in `tests/check_worker_gpu.py`; and both
file-video memory paths plus Live in `tests/check_worker_video.py`.
