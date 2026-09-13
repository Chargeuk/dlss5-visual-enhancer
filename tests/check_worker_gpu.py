"""Kill only this test's own native worker and verify recovery on the real GPU."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.core.jobs import JobController, active_job
from src.core.runtime import DLSSFrameSession, prepare_runtime, resolve_native_settings, resolve_upscaling_mode
from src.core.neural_bridge import BRIDGE_MANAGER
from src.core.neural_worker import WORKER, WorkerLost
from src.neural_rendering.video.models import ConversionOptions
from src.core.gpu_selection import resolve_runtime_ai_gpu
from src.settings.storage import processing_gpu_settings

prepared=prepare_runtime()
gpu=resolve_runtime_ai_gpu(prepared.gpus,prepared.runtime_bundle,processing_gpu_settings()[0])
factor,mode=resolve_upscaling_mode(1)
kwargs=dict(input_width=128,input_height=96,output_width=128,output_height=96,frame_count=2,
            warmup_frames=0,factor=factor,mode=mode,native_settings=resolve_native_settings(ConversionOptions(nr_passes=2)),
            gpu=gpu,runtime_bundle=prepared.runtime_bundle)
pixels=np.full((96,128,4),110,np.uint8); pixels[...,3]=180
main_pid=os.getpid()
with active_job() as controller:
    session=DLSSFrameSession(**kwargs,controller=controller)
    try:
        session.process(index=0,rgba=pixels,reset=True,pts=0)
        first_pid=WORKER.process.pid
        WORKER.process.kill(); WORKER.process.wait(timeout=10)
        result,_=session.process(index=1,rgba=pixels,reset=True,pts=1)
        assert WORKER.process.pid!=first_pid and os.getpid()==main_pid
        assert np.all(result[...,3]==180)
        session.close()
        assert session.completed_frames==2 and session.diagnostics.feature_evaluations==4
        assert session.structured_status()['worker_restarts']==1
        assert BRIDGE_MANAGER._initialized_ordinal is None
    finally: session.abort()
print('PASS native worker killed: still frame retried, server-side Python PID unchanged',flush=True)

with active_job() as controller:
    session=DLSSFrameSession(**kwargs,controller=controller,recover_reset_frames=False)
    try:
        session.process(index=0,rgba=pixels,reset=True,pts=0)
        WORKER.process.kill(); WORKER.process.wait(timeout=10)
        try: session.process(index=1,rgba=pixels,reset=False,pts=1)
        except WorkerLost: pass
        else: raise AssertionError('Temporal history must be replayed after a crash')
    finally: session.abort()
print('PASS temporal failure requests replay instead of silently dropping history',flush=True)

with active_job() as controller:
    session=DLSSFrameSession(**dict(kwargs,frame_count=1),controller=controller)
    session.process(index=0,rgba=pixels,reset=True,pts=0); session.close()
print('PASS next queued logical job can use a fresh worker',flush=True)
WORKER.stop()
