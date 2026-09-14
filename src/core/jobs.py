from __future__ import annotations

import asyncio
import atexit
import logging
import os
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager, asynccontextmanager, suppress
from contextvars import ContextVar
from typing import Iterator


class Cancelled(RuntimeError):
    pass


class JobController:
    """Own cancellation state and subprocesses for one render."""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._processes: list[subprocess.Popen] = []

    def register(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.append(process)
            cancelled = self.cancel.is_set()
        if cancelled and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def unregister(self, process: subprocess.Popen) -> None:
        with self._lock:
            if process in self._processes:
                self._processes.remove(process)

    def stop(self) -> None:
        self.cancel.set()
        self.terminate_processes()

    def terminate_processes(self) -> None:
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass


_ACTIVE_LOCK = threading.Lock()
_ACTIVE: JobController | None = None
_WAITING = deque()
_IDLE_TIMER = None
_STOPPING_WORKER = False
IDLE_WORKER_SECONDS = 10.0
_JOB_CONTEXT = ContextVar("render_job_controller", default=None)


def current_job_controller():
    return _JOB_CONTEXT.get()


@contextmanager
def use_job_controller(controller: JobController):
    token = _JOB_CONTEXT.set(controller)
    try:
        yield
    finally:
        _JOB_CONTEXT.reset(token)


def _cancel_idle_timer_locked():
    global _IDLE_TIMER
    timer, _IDLE_TIMER = _IDLE_TIMER, None
    if timer is not None:
        timer.cancel()


def _stop_idle_worker():
    from .memory_cleanup import stop_idle_worker
    stop_idle_worker()


def _cleanup_job_memory():
    from .memory_cleanup import cleanup_job_memory
    cleanup_job_memory()


def _schedule_idle_timer_locked():
    global _IDLE_TIMER
    # Child video/live pipelines have their own local queue, but the parent
    # exclusively owns worker lifetime and the application-wide idle timer.
    if (_ACTIVE is not None or _WAITING or _IDLE_TIMER is not None or _STOPPING_WORKER
            or os.environ.get('MERSERK_NEURAL_WORKER') == '1'):
        return

    def expired():
        global _IDLE_TIMER, _STOPPING_WORKER
        with _ACTIVE_LOCK:
            # cancel() alone cannot stop a callback that has already begun.
            if _IDLE_TIMER is not timer:
                return
            _IDLE_TIMER = None
            if _ACTIVE is not None or _WAITING:
                return
            _STOPPING_WORKER = True
        # Requests can still enqueue (including ASGI calls) during shutdown,
        # but none can claim a slot and start using the worker being stopped.
        try:
            _stop_idle_worker()
        except Exception:
            logging.getLogger(__name__).exception('Idle neural worker shutdown failed')
        finally:
            with _ACTIVE_LOCK:
                _STOPPING_WORKER = False

    timer = threading.Timer(IDLE_WORKER_SECONDS, expired)
    timer.daemon = True
    _IDLE_TIMER = timer
    timer.start()


def _cancel_idle_timer():
    with _ACTIVE_LOCK:
        _cancel_idle_timer_locked()


atexit.register(_cancel_idle_timer)


def _enqueue(controller):
    ticket = object()
    with _ACTIVE_LOCK:
        _cancel_idle_timer_locked()
        _WAITING.append(ticket)
    return ticket


def _try_claim(ticket, controller):
    global _ACTIVE
    with _ACTIVE_LOCK:
        if controller.cancel.is_set():
            raise Cancelled("Queued render cancelled.")
        if not _STOPPING_WORKER and _ACTIVE is None and _WAITING and _WAITING[0] is ticket:
            _WAITING.popleft()
            _ACTIVE = controller
            return True
    return False


def _position(ticket):
    with _ACTIVE_LOCK:
        return _WAITING.index(ticket) + 1


def _leave(ticket, controller, claimed):
    global _ACTIVE
    try:
        if claimed:
            try:
                controller.terminate_processes()
            finally:
                # Keep ownership until cleanup finishes, including failures and
                # cancellations. Queued jobs must not lose their allocations.
                try:
                    _cleanup_job_memory()
                except Exception:
                    logging.getLogger(__name__).exception('Post-job memory cleanup failed')
    finally:
        with _ACTIVE_LOCK:
            if claimed and _ACTIVE is controller:
                _ACTIVE = None
            with suppress(ValueError):
                _WAITING.remove(ticket)
            _schedule_idle_timer_locked()


def _report_wait(position):
    # Gradio associates progress with the current request, not another GUI.
    with suppress(Exception):
        import gradio as gr
        gr.Progress()(0, desc=f"Waiting for GPU — queue position {position}")


@contextmanager
def active_job(controller: JobController | None = None) -> Iterator[JobController]:
    """Wait in FIFO order; cancellation removes only this queued request."""
    controller = controller or current_job_controller() or JobController()
    ticket = _enqueue(controller)
    claimed = False
    previous = None
    try:
        while not claimed:
            claimed = _try_claim(ticket, controller)
            if not claimed:
                position = _position(ticket)
                if position != previous:
                    _report_wait(position)
                    previous = position
                controller.cancel.wait(.1)
        yield controller
    finally:
        _leave(ticket, controller, claimed)


@asynccontextmanager
async def queued_socket_job(websocket, controller, setup):
    """Use the same FIFO without blocking ASGI; watch queued disconnects."""
    from starlette.websockets import WebSocketDisconnect
    ticket = _enqueue(controller)
    claimed = False
    disconnected = asyncio.create_task(websocket.receive())
    next_notice = 0
    deadline = time.monotonic() + setup.get("timeout_seconds", 600)
    try:
        while not claimed:
            if disconnected.done():
                message = disconnected.result()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))
                raise ValueError("Wait for server readiness before sending frames.")
            claimed = _try_claim(ticket, controller)
            if not claimed:
                now = time.monotonic()
                if setup.get("queue_status", False):
                    if now >= next_notice:
                        await asyncio.wait_for(websocket.send_json(dict(
                            type="queued", position=_position(ticket))), 10)
                        next_notice = now + min(5, setup.get("timeout_seconds", 600) / 3)
                elif now >= deadline:
                    raise TimeoutError("Timed out waiting for the GPU queue.")
                await asyncio.sleep(.1)
        disconnected.cancel()
        with suppress(asyncio.CancelledError):
            await disconnected
        yield controller
    finally:
        disconnected.cancel()
        try:
            with suppress(BaseException):
                await disconnected
        finally:
            cleanup = asyncio.create_task(asyncio.to_thread(_leave, ticket, controller, claimed))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # The cleanup thread retains queue ownership even if the
                # disconnected request is cancelled again while it is finishing.
                raise


def cancel_active_job() -> str:
    with _ACTIVE_LOCK:
        controller = _ACTIVE
    if controller is None:
        return "No render is running."
    controller.stop()
    return "Stop requested; incomplete output will be removed and completed batch files retained."


def drain_text(stream, lines: list[str]) -> None:
    for raw in iter(stream.readline, b""):
        lines.append(raw.decode("utf-8", "replace").rstrip())


class BoundedLogBuffer:
    """Keep diagnostic evidence and a bounded tail instead of every frame log."""

    _IMPORTANT = (
        "profile applied",
        "bridge",
        "ngx",
        "stream source",
        "optimal settings",
        "complete:",
        "failed",
        "error",
        "exception",
    )

    def __init__(self, max_tail: int = 500, max_important: int = 100) -> None:
        self._tail: deque[str] = deque(maxlen=max_tail)
        self._important: list[str] = []
        self._max_important = max_important
        self._seen = 0
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self._seen += 1
            self._tail.append(line)
            lowered = line.casefold()
            if (
                len(self._important) < self._max_important
                and any(marker.casefold() in lowered for marker in self._IMPORTANT)
            ):
                self._important.append(line)

    def snapshot(self) -> list[str]:
        with self._lock:
            important = list(self._important)
            tail = list(self._tail)
        seen: set[str] = set()
        return [line for line in [*important, *tail] if not (line in seen or seen.add(line))]

    @property
    def dropped_lines(self) -> int:
        with self._lock:
            return max(0, self._seen - len(self._tail))


def drain_bounded_text(stream, buffer: BoundedLogBuffer) -> None:
    for raw in iter(stream.readline, b""):
        buffer.append(raw.decode("utf-8", "replace").rstrip())
