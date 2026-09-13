import sys
import unittest
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.core import neural_worker as module
from src.core.jobs import JobController, Cancelled
from src.core.neural_bridge import BridgeSessionDiagnostics


class FakeWorker:
    generation=0
    process=None
    def __init__(self): self.opens=0; self.calls=0; self.failures=0; self.frames=0
    def stop(self): self.process=None
    def call(self,request,controller,**kwargs):
        if controller.cancel.is_set(): raise Cancelled('cancelled')
        if request['op']=='open':
            self.opens+=1; self.frames=0
            self.generation+=1
            self.process=SimpleNamespace(poll=lambda:None,pid=1234)
        elif request['op']=='process':
            self.calls+=1
            if self.failures:
                self.failures-=1; self.stop(); raise module.WorkerLost(module.RECOVERY_CODE)
            self.frames+=1
        return dict(diagnostics=BridgeSessionDiagnostics(gpu_mode=True,memory_path='shared',frames=self.frames,feature_evaluations=self.frames),
                    timings={},status={},logs=[])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker=FakeWorker()
        patcher=patch.object(module,'WORKER',self.worker); patcher.start(); self.addCleanup(patcher.stop)
        self.kwargs=dict(output_width=64,output_height=64,frame_count=1,native_settings={},
                         runtime_bundle={},factor=1,mode={},controller=JobController())
        self.pixels=np.zeros((64,64,4),np.uint8)

    def test_reset_frame_retries_once_and_next_job_remains_usable(self):
        self.worker.failures=1
        session=module.RemoteFrameSession(**self.kwargs)
        try:
            session.process(index=0,rgba=self.pixels,reset=True,pts=0)
            self.assertEqual((self.worker.opens,self.worker.calls),(2,2))
            session.close()
        finally: session.abort()
        self.assertEqual(session.completed_frames,1)

    def test_repeated_failure_is_bounded(self):
        self.worker.failures=2
        session=module.RemoteFrameSession(**self.kwargs)
        try:
            with self.assertRaises(module.WorkerLost):
                session.process(index=0,rgba=self.pixels,reset=True,pts=0)
            self.assertEqual((self.worker.opens,self.worker.calls),(2,2))
        finally: session.abort()

    def test_temporal_caller_requests_replay_instead_of_resetting_history(self):
        self.worker.failures=1
        session=module.RemoteFrameSession(**self.kwargs,recover_reset_frames=False)
        try:
            with self.assertRaises(module.WorkerLost):
                session.process(index=0,rgba=self.pixels,reset=True,pts=0)
            self.assertEqual(self.worker.opens,1)
        finally: session.abort()

    def test_cancelled_request_is_not_retried(self):
        session=module.RemoteFrameSession(**self.kwargs)
        self.kwargs['controller'].stop()
        try:
            with self.assertRaises(Cancelled):
                session.process(index=0,rgba=self.pixels,reset=True,pts=0)
            self.assertEqual(self.worker.opens,1)
        finally: session.abort()

    def test_video_worker_retries_only_fatal_failures(self):
        @dataclass
        class Result:
            output_path: str
        lost=module.WorkerLost('died')
        attempts=[]
        def convert(request,*args,**kwargs):
            attempts.append(request)
            target=Path(request['output_dir'])/'result.mp4'
            target.write_bytes(b'video')
            if len(attempts)==1: raise lost
            return {'result':Result(str(target))}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.worker,'call',side_effect=convert) as call:
                result=module.worker_video('input',None,directory,JobController(),None)
                self.assertEqual(Path(result.output_path).read_bytes(),b'video')
                self.assertEqual(call.call_count,2)
                self.assertEqual([p.name for p in Path(directory).iterdir()],['result.mp4'])
            with patch.object(self.worker,'call',side_effect=ValueError('bad options')) as call:
                with self.assertRaises(ValueError): module.worker_video('input',None,directory,JobController(),None)
                self.assertEqual(call.call_count,1)


if __name__=='__main__': unittest.main()
