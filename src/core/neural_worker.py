"""Restartable host-frame neural worker. Only trusted anonymous pipes use pickle."""
from __future__ import annotations

import atexit
import contextlib
import logging
import os
import pickle
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

from .jobs import Cancelled
from .neural_bridge import NeuralBridgeError, NeuralBridgePoisonedError
from .worker_job import own_process

RECOVERY_CODE = "NEURAL_WORKER_RESTARTED"


class WorkerLost(NeuralBridgeError):
    """The worker was discarded; temporal callers must replay their inputs."""


def send(pipe, value):
    payload = pickle.dumps(value, protocol=5)
    pipe.write(struct.pack('<Q', len(payload)))
    pipe.write(payload)
    pipe.flush()


def receive(pipe):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            block = pipe.read(size - len(data))
            if not block:
                raise EOFError('Neural worker exited')
            data.extend(block)
        return data
    length, = struct.unpack('<Q', exact(8))
    if length > 32 * 1024 * 1024:
        raise ValueError('Invalid worker control payload')
    return pickle.loads(exact(length))


class Worker:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.responses = None
        self.generation = 0
        self.close_job = None
        self.scratch = None

    def stop(self):
        with self.lock:
            process, self.process = self.process, None
            if process is None:
                if self.scratch is not None:
                    self.scratch.cleanup(); self.scratch = None
                return
            if self.close_job is not None:
                self.close_job(); self.close_job = None
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            for stream in (process.stdin, process.stdout):
                with contextlib.suppress(Exception): stream.close()
            if self.scratch is not None:
                self.scratch.cleanup(); self.scratch = None

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.stop()
        root = Path(__file__).resolve().parents[2]
        (root/'temp').mkdir(exist_ok=True)
        # NVIDIA can retain extracted DLL cache files after the child exits.
        # Best-effort cache cleanup must never prevent replacing a dead worker.
        self.scratch = tempfile.TemporaryDirectory(prefix='neural-worker-',dir=root/'temp',ignore_cleanup_errors=True)
        env = dict(os.environ, MERSERK_NEURAL_WORKER='1', PYTHONIOENCODING='utf-8',
                   TEMP=self.scratch.name,TMP=self.scratch.name)
        process = subprocess.Popen([sys.executable, str(root/'tools'/'neural_worker.py')],
            cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.process = process
        try: self.close_job = own_process(process)
        except BaseException:
            self.stop()
            raise
        self.generation += 1
        responses = self.responses = queue.Queue()
        def reader():
            try:
                while True: responses.put(receive(process.stdout))
            except (EOFError, OSError, ValueError) as exc:
                responses.put(exc)
        threading.Thread(target=reader, daemon=True, name='neural-worker-replies').start()

    def call(self, command, controller, timeout=200, progress=None):
        with self.lock:
            if controller.cancel.is_set():
                raise Cancelled('Render cancelled.')
            self.start()
            try:
                send(self.process.stdin, command)
                deadline = time.monotonic() + timeout
                while True:
                    if controller.cancel.is_set():
                        self.stop()
                        raise Cancelled('Render cancelled; its neural worker was stopped.')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Neural worker did not respond before its deadline')
                    try: response = self.responses.get(timeout=.1)
                    except queue.Empty: continue
                    if isinstance(response, Exception): raise response
                    if 'progress' in response:
                        deadline = time.monotonic() + timeout
                        if progress: progress(*response['progress'])
                        continue
                    if response.get('fatal'):
                        raise WorkerLost(response['error'])
                    if 'error' in response:
                        raise NeuralBridgeError(response['error'])
                    return response
            except (EOFError, OSError, TimeoutError, WorkerLost) as exc:
                logging.getLogger(__name__).warning('Discarding neural worker after failure: %s', exc)
                self.stop()
                raise WorkerLost(f'{RECOVERY_CODE}: {exc}') from exc
            except BaseException:
                self.stop()
                raise


WORKER = Worker()
atexit.register(WORKER.stop)


class RemoteFrameSession:
    """Keep Python/web state alive while the native DLL lives in a child process."""
    def __init__(self, **kwargs):
        if kwargs.get('cuda_video'):
            raise ValueError('CUDA video sessions must be created inside the neural worker.')
        self.controller = kwargs.pop('controller')
        self.recover_reset_frames = kwargs.pop('recover_reset_frames', True)
        self.expected = kwargs.get('frame_count')
        self.sent = 0
        self.next_index = None
        self.retries = 0
        self.closed = False
        self.completed_frames = None
        self.kwargs = kwargs
        self.token = uuid.uuid4().hex
        self.native_settings = dict(kwargs['native_settings'])
        self.composition_mask = kwargs.get('composition_mask')
        self.runtime_bundle = kwargs['runtime_bundle']
        self.factor, self.mode = kwargs['factor'], kwargs['mode']
        self.output_width = self.render_width = kwargs['output_width']
        self.output_height = self.render_height = kwargs['output_height']
        self.minimum_width = self.minimum_height = 64
        self.maximum_width = self.maximum_height = 16384
        self.setup_result = 1
        self._status = {}
        self._logs = []
        self._prior_frames = self._prior_evaluations = self._prior_resets = 0
        self.memory = SharedMemory(create=True, size=self.output_width*self.output_height*8)
        self.input = np.ndarray((self.output_height,self.output_width,4), np.uint8, buffer=self.memory.buf)
        self.output = np.ndarray(self.input.shape, np.uint8, buffer=self.memory.buf, offset=self.input.nbytes)
        try:
            try: self._open()
            except WorkerLost:
                if not self.recover_reset_frames: raise
                self.retries += 1
                self._open()
        except BaseException:
            self._release_memory()
            raise

    def _open(self):
        kwargs = dict(self.kwargs, frame_count=None)
        self._update(WORKER.call(dict(op='open',token=self.token,kwargs=kwargs,memory=self.memory.name), self.controller))
        self.generation = WORKER.generation

    def _update(self, response):
        self.diagnostics = response['diagnostics']
        self.diagnostics.frames += self._prior_frames
        self.diagnostics.feature_evaluations += self._prior_evaluations
        self.diagnostics.scene_resets += self._prior_resets
        self.process_timings = response['timings']
        self.bridge_status = response['status']
        self._status = dict(self.bridge_status)
        self._logs = response['logs']

    @property
    def bridge_logs(self): return list(self._logs)
    @property
    def logs(self): return self.bridge_logs
    @property
    def bridge_log_dropped_lines(self): return 0

    def structured_status(self):
        return dict(self._status, **self.diagnostics.as_dict(), worker_restarts=self.retries,
                    worker_pid=WORKER.process.pid if WORKER.process is not None else None)

    def process(self, *, index, rgba, reset, pts, motion=None, output_buffer=None):
        if self.closed: raise RuntimeError('Neural session is closed.')
        if rgba.dtype != np.uint8 or rgba.shape != self.input.shape:
            raise ValueError('Neural input must be RGBA8 at the negotiated dimensions.')
        if self.next_index is not None and index != self.next_index:
            raise ValueError('Neural frames must have consecutive indices.')
        if output_buffer is not None and (output_buffer.shape != rgba.shape or output_buffer.dtype != np.uint8):
            raise ValueError('Invalid neural output buffer.')
        np.copyto(self.input, rgba)
        command = dict(op='process',token=self.token,index=index,reset=reset,pts=pts,
                       settings=self.native_settings)
        try:
            if WORKER.generation != self.generation or WORKER.process is None or WORKER.process.poll() is not None:
                WORKER.stop()
                raise WorkerLost(RECOVERY_CODE)
            response = WORKER.call(command,self.controller)
        except WorkerLost:
            # Only reset frames are independently reproducible. A temporal stream
            # must replay from its original inputs to rebuild both VSR/NR history.
            if not self.recover_reset_frames or not reset or self.retries:
                raise
            self.retries += 1
            self._prior_frames = self.diagnostics.frames
            self._prior_evaluations = self.diagnostics.feature_evaluations
            self._prior_resets = self.diagnostics.scene_resets
            self._open()
            response = WORKER.call(command,self.controller)
        self._update(response)
        self.sent += 1
        self.next_index = index+1
        result = self.output.copy() if output_buffer is None else output_buffer
        if output_buffer is not None: np.copyto(result,self.output)
        return result, pts

    def update_composition_mask(self, selection, feather):
        self._update(WORKER.call(dict(op='mask',token=self.token,selection=selection,feather=feather),self.controller))
        self.composition_mask = selection
        self.kwargs['composition_mask'] = selection
        self.kwargs['native_settings']['mask_feather'] = feather

    def _release_memory(self):
        self.input = self.output = None
        self.memory.close()
        self.memory.unlink()

    def _finish(self, abort):
        if self.closed: return
        try:
            if WORKER.process is not None and WORKER.process.poll() is None and WORKER.generation == self.generation:
                if self.controller.cancel.is_set():
                    WORKER.stop()
                else:
                    # Closing must not cause a lost worker to be resurrected.
                    self._update(WORKER.call(dict(op='close',token=self.token,abort=abort), self.controller))
        finally:
            self.closed = True
            self._release_memory()

    def close(self):
        if self.closed: return
        if self.controller.cancel.is_set():
            self.abort(); raise Cancelled('Render cancelled.')
        if self.expected is not None and self.sent != self.expected:
            self.abort(); raise RuntimeError(f'Neural Rendering processed {self.sent} of {self.expected} frames.')
        if not self.sent:
            self.abort(); raise RuntimeError('The input contains no frames.')
        self._finish(False)
        self.completed_frames = self.sent

    def abort(self):
        with contextlib.suppress(WorkerLost): self._finish(True)


def worker_video(source, options, output_dir, controller, progress):
    from dataclasses import replace
    from .disk_paths import prepare_output_dir
    destination = prepare_output_dir(output_dir)
    for attempt in range(2):
        try:
            with tempfile.TemporaryDirectory(prefix='.nr-attempt-',dir=destination) as directory:
                result = WORKER.call(dict(op='video',source=source,options=options,output_dir=directory),
                                     controller, progress=progress)['result']
                rendered = Path(result.output_path).resolve()
                if rendered.parent != Path(directory).resolve():
                    raise ValueError('Worker returned a video outside its attempt directory.')
                target = destination / rendered.name
                # Atomic publish with no replacement of another user's file.
                if os.name == 'nt': os.rename(rendered,target)
                else:
                    os.link(rendered,target); rendered.unlink()
                return replace(result,output_path=str(target))
        except WorkerLost:
            if attempt: raise
            if progress: progress(0, 'Restarting neural worker and retrying video from the beginning')
