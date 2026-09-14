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

## Memory cleanup and idle shutdown

Every completed, failed or cancelled GPU job releases disposable memory before
handing its queue slot to the next job. Existing session teardown frees frame
buffers, CUDA masks and feature-owned surfaces. The supervisor additionally asks
the existing neural worker to collect unused memory, collects its own unused
objects, closes warmed runtime file mappings, and clears unused PyTorch CUDA
allocator blocks if PyTorch is already loaded and initialized. Cleanup never
imports or initializes a GPU framework solely to empty its cache. Small runtime
metadata remains cached. A cleanup failure is logged and cannot strand the queue;
a failed worker cleanup discards the child without invalidating completed output.

Pending GUI thumbnails move from the bounded RAM cache to the existing temporary
preview storage and remain available for delivery. If staging fails, the pending
preview is retained rather than lost. This does not create image files for VTS
no-save calls; they do not use that preview cache. Returned images and active
playback remain owned by their consumers and are not forcibly invalidated.

After cleanup, a single 10-second timer starts only when no job is running or
waiting. Enqueuing any new request cancels it. The callback checks its identity
and queue state under the queue lock, reserves shutdown, and stops the reusable
neural worker outside that lock while claiming a job is excluded. Enqueuing stays
responsive throughout shutdown. It then discards the timer. Requests arriving during
shutdown wait for shutdown to complete and automatically initialize a fresh
worker. Child pipelines do not create independent idle timers. The separate Live
worker stays alive while it owns active playback; its render stage still performs
per-job cleanup, and stopping Live releases that process.

CUDA contexts, runtime libraries and driver-managed allocations can remain while
the reusable worker is alive; its exit releases the remaining process-owned
resources. The web process, displayed results and operating-system file caches
can still use RAM. This is not a promise of zero application memory while idle.

Regression tests: `tests/test_memory_cleanup.py` covers cancellation, FIFO
ownership, stale timer callbacks, simultaneous arrival/shutdown, asynchronous
cleanup, cleanup failures and preview preservation. Run the opt-in GPU check
`tests/check_worker_idle.py` to verify reuse, the real 10-second idle shutdown,
and automatic worker initialization on the following job.
