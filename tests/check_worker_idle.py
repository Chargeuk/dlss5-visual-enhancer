"""Opt-in real GPU check: reuse, idle release, and automatic cold start."""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.core import jobs
from src.core.neural_worker import WORKER
from src.core.runtime import DLSSFrameSession, prepare_runtime, resolve_native_settings, resolve_upscaling_mode
from src.core.gpu_selection import resolve_runtime_ai_gpu
from src.neural_rendering.video.models import ConversionOptions
from src.settings.storage import processing_gpu_settings


def main():
    prepared = prepare_runtime()
    gpu = resolve_runtime_ai_gpu(prepared.gpus, prepared.runtime_bundle, processing_gpu_settings()[0])
    factor, mode = resolve_upscaling_mode(1)
    kwargs = dict(input_width=128, input_height=96, output_width=128, output_height=96,
                  frame_count=1, warmup_frames=0, factor=factor, mode=mode,
                  native_settings=resolve_native_settings(ConversionOptions(nr_passes=1)),
                  gpu=gpu, runtime_bundle=prepared.runtime_bundle)
    pixels = np.full((96, 128, 4), 110, np.uint8)
    pixels[..., 3] = 180
    def render():
        with jobs.active_job() as controller:
            session = DLSSFrameSession(**kwargs, controller=controller)
            try:
                result, _ = session.process(index=0, rgba=pixels, reset=True, pts=0)
                session.close()
                assert np.all(result[..., 3] == 180)
                process = WORKER.process
            finally:
                session.abort()
        assert not prepared._mappings
        return process
    server_pid = os.getpid()
    try:
        first = render()
        timer = jobs._IDLE_TIMER
        assert first.poll() is None and timer is not None
        second = render()
        assert second is first and timer is not jobs._IDLE_TIMER
        print('PASS consecutive jobs reuse worker; cleanup preserves output and releases warmed mappings', flush=True)
        started = time.monotonic()
        while time.monotonic() - started < 20:
            with jobs._ACTIVE_LOCK:
                stopped = WORKER.process is None and not jobs._STOPPING_WORKER
            if stopped: break
            time.sleep(.1)
        elapsed = time.monotonic() - started
        assert stopped and elapsed >= 9.5, elapsed
        assert first.poll() is not None and os.getpid() == server_pid
        print(f'PASS idle worker exited after {elapsed:.1f}s; supervising Python remained alive', flush=True)
        third = render()
        assert third.pid != first.pid
        print('PASS next job automatically starts a fresh working neural worker', flush=True)
    finally:
        jobs._cancel_idle_timer()
        WORKER.stop()


if __name__ == '__main__': main()
