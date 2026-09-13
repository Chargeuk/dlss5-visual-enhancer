import io
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from src.neural_rendering import sequence as module
from src.core.jobs import JobController


def png(color, size=(128, 96), mode='RGBA'):
    with Image.new(mode, size, color) as image, io.BytesIO() as buffer:
        image.save(buffer, format='PNG')
        return buffer.getvalue()


class Neural:
    instances = []
    def __init__(self, **kwargs):
        self.settings = kwargs
        self.resets = []
        self.diagnostics = SimpleNamespace(feature_evaluations=0)
        self.closed = False
        self.bridge_logs = []
        self.instances.append(self)
    def process(self, *, index, rgba, reset, pts):
        assert index == len(self.resets) and pts == index
        self.resets.append(reset)
        self.diagnostics.feature_evaluations += self.settings['native_settings']['nr_passes']
        return rgba.copy(), pts
    def structured_status(self): return {'temporal_stabilization': {'stabilized_frames': self.resets.count(False)}}
    def close(self): self.closed = True
    def abort(self): self.closed = True


class VSR:
    instances = []
    def __init__(self, width, height, ow, oh, *args, **kwargs):
        self.input_size, self.target = (width,height), (ow,oh)
        self.completed_frames = 0
        self.closed = False
        self.instances.append(self)
    def process_frame(self, rgba):
        self.completed_frames += 1
        with Image.fromarray(rgba) as image, image.resize(self.target) as result:
            return result.tobytes()
    def close(self, **kwargs): self.closed = True


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        Neural.instances, VSR.instances = [], []
        self.setup = dict(version=1,width=128,height=96,target_width=192,target_height=144,
                          channels=4,frame_count=3,parameters={'nr_passes':2,'shimmer_suppression':.7})
        for name,value in [('DLSSFrameSession',Neural),('RTXVideoSession',VSR),
                           ('prepare_runtime',lambda:SimpleNamespace(gpus=[],runtime_bundle={})),
                           ('resolve_runtime_ai_gpu',lambda *a:{'display_name':'test'}),
                           ('processing_gpu_settings',lambda:('auto','auto')),
                           ('probe_capabilities',lambda *a,**k:None),('verify_feature_18',lambda *a:None)]:
            patcher=patch.object(module,name,value); patcher.start(); self.addCleanup(patcher.stop)

    def test_persistent_sessions_history_cuts_passes_and_alpha(self):
        stream=module.EnhancementSequence(module.validate_setup(self.setup),JobController())
        stream.open()
        for index,color in enumerate([(20,20,20,180),(22,22,22,180),(245,245,245,180)]):
            data,cut=stream.push(png(color),index)
            self.assertEqual(cut,index==2)
            with Image.open(io.BytesIO(data)) as result:
                self.assertEqual(result.size,(192,144))
                self.assertEqual(result.getpixel((0,0))[3],180)
        stream.close()
        self.assertEqual(len(Neural.instances),1)
        self.assertEqual(len(VSR.instances),1)
        self.assertEqual(Neural.instances[0].resets,[True,False,True])
        self.assertEqual(stream.stats['neural_evaluations'],6)
        self.assertEqual(stream.stats['scene_cuts'],1)
        self.assertEqual(stream.stats['vsr_frames'],3)
        self.assertTrue(Neural.instances[0].closed and VSR.instances[0].closed)

    def test_neural_only_skips_vsr_and_new_sequence_starts_fresh(self):
        for _ in range(2):
            setup=dict(self.setup,target_width=128,target_height=96,frame_count=1)
            stream=module.EnhancementSequence(module.validate_setup(setup),JobController())
            stream.open(); stream.push(png((20,20,20,255)),0); stream.close()
        self.assertFalse(VSR.instances)
        self.assertEqual([session.resets for session in Neural.instances],[[True],[True]])

    def test_rejects_hdr_outer_iterations_bad_dimensions_and_wrong_png(self):
        for override in [dict(parameters={'iterations':2}),dict(parameters={'nr_passes':5}),
                         dict(channels=2),dict(target_width=32),dict(width=True)]:
            with self.subTest(override=override),self.assertRaises(ValueError):
                module.validate_setup(dict(self.setup,**override))
        stream=module.EnhancementSequence(self.setup,JobController())
        with self.assertRaises(ValueError): stream.push(png('red',mode='RGB'),0)
        with self.assertRaises(ValueError): stream.push(png('red'),1)


class FakeSequence:
    instances=[]
    block=False
    fail=False
    def __init__(self,setup,controller):
        self.controller=controller; self.stats={'frames':0}
        self.closed=threading.Event(); self.entered=threading.Event(); self.threads=[]
        self.instances.append(self)
    def open(self): self.threads.append(threading.get_ident())
    def push(self,data,index):
        self.threads.append(threading.get_ident()); self.entered.set()
        if self.fail: raise RuntimeError('test renderer failure')
        if self.block: self.controller.cancel.wait(3)
        self.stats['frames']+=1
        return bytes(data),False
    def close(self,**kwargs):
        self.threads.append(threading.get_ident()); self.closed.set()


class SocketTests(unittest.TestCase):
    def setUp(self):
        FakeSequence.instances=[]; FakeSequence.block=False; FakeSequence.fail=False
        patcher=patch.object(module,'EnhancementSequence',FakeSequence); patcher.start(); self.addCleanup(patcher.stop)
        self.client=TestClient(Starlette(routes=[WebSocketRoute('/vts/enhance_sequence',module.enhancement_socket)]))
        self.setup=dict(version=1,width=128,height=96,target_width=128,target_height=96,channels=4,frame_count=2)
        self.png=png((50,60,70,180))
    def connect(self): return self.client.websocket_connect('/vts/enhance_sequence')
    def send_frame(self,socket,index):
        socket.send_json(dict(type='frame',index=index,bytes=len(self.png)))
        socket.send_bytes(self.png[:10]); socket.send_bytes(self.png[10:])

    def test_one_output_per_input_same_worker_and_reusable_slot(self):
        for _ in range(2):
            with self.connect() as socket:
                socket.send_json(self.setup)
                self.assertEqual(socket.receive_json()['output_count'],2)
                for i in range(2):
                    self.send_frame(socket,i)
                    header=socket.receive_json()
                    self.assertEqual((header['type'],header['index']),('enhanced',i))
                    self.assertEqual(socket.receive_bytes(),self.png)
                socket.send_json(dict(type='end'))
                self.assertEqual(socket.receive_json()['type'],'done')
                self.assertTrue(FakeSequence.instances[-1].closed.is_set())
                self.assertEqual(len(set(FakeSequence.instances[-1].threads)),1)

    def test_disconnect_during_processing_releases_slot_after_worker_finishes(self):
        FakeSequence.block=True
        with self.connect() as socket:
            socket.send_json(self.setup); socket.receive_json(); self.send_frame(socket,0)
            self.assertTrue(FakeSequence.instances[-1].entered.wait(2))
        self.assertTrue(FakeSequence.instances[-1].closed.wait(3))
        FakeSequence.block=False
        with self.connect() as socket:
            socket.send_json(self.setup); self.assertEqual(socket.receive_json()['type'],'ready')

    def test_busy_client_cannot_cancel_owner(self):
        with self.connect() as first:
            first.send_json(self.setup); first.receive_json()
            with self.connect() as second:
                second.send_json(self.setup)
                self.assertIn('already running',second.receive_json()['message'])
            self.assertFalse(FakeSequence.instances[0].controller.cancel.is_set())

    def test_invalid_order_and_renderer_failure_cleanup(self):
        for fail in (False,True):
            FakeSequence.fail=fail
            with self.connect() as socket:
                socket.send_json(self.setup); socket.receive_json()
                self.send_frame(socket,0 if fail else 9)
                self.assertEqual(socket.receive_json()['type'],'error')
                self.assertTrue(FakeSequence.instances[-1].closed.is_set())


if __name__=='__main__': unittest.main()
