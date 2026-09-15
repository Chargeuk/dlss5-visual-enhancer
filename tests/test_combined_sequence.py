import io
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient

import test_temporal_sequence as temporal
from src.neural_rendering import sequence
from src.frame_interpolation import stream as interpolation
from src.core.jobs import JobController


class NativeInterpolation:
    instances = []
    def __init__(self, width, height, count, generated, controller):
        self.previous = None
        self.closed = False
        self.instances.append(self)
    def process_frame(self, rgba, motion, timestamp, *, reset):
        assert motion.shape == (*rgba.shape[:2], 2)
        frames = [] if reset else [np.rint((self.previous.astype(float)+rgba)/2).astype(np.uint8)]
        self.previous = rgba.copy()
        return frames
    def close(self): self.closed = True


class CombinedTests(unittest.TestCase):
    def setUp(self):
        temporal.ProcessingTests.setUp(self)
        NativeInterpolation.instances=[]
        for name,value in [('DirectDLSSGSession',NativeInterpolation),
                           ('probe_frame_interpolation_capabilities',lambda:SimpleNamespace(available=True))]:
            p=patch.object(interpolation,name,value); p.start(); self.addCleanup(p.stop)
        original=temporal.Neural.process
        def enhanced(session, **kwargs):
            rgba,pts=original(session,**kwargs)
            rgba[...,:3] = np.minimum(rgba[...,:3].astype(int)+10,255).astype(np.uint8)
            return rgba,pts
        p=patch.object(temporal.Neural,'process',enhanced); p.start(); self.addCleanup(p.stop)
        self.setup.update(version=2,target_width=128,target_height=96,
                          enable_frame_interpolation=True,interpolation_multiplier=2)

    def test_all_multipliers_use_enhanced_originals_and_keep_neural_evaluation_count(self):
        for multiplier in (2,3,4,8):
            stream=sequence.CombinedEnhancementSequence(sequence.validate_setup(dict(self.setup,interpolation_multiplier=multiplier)),JobController())
            stream.open()
            outputs=[]
            try:
                for i,color in enumerate((20,36,52)):
                    frames,cut=stream.push_frames(temporal.png((color,color,color,180)),i)
                    self.assertFalse(cut)
                    self.assertLessEqual(len(frames),multiplier)
                    outputs.extend(int(frame[0,0,0]) for frame in frames)
                    self.assertTrue(all(np.all(frame[...,3]==180) for frame in frames))
            finally: stream.close()
            positions=[Fraction(3,8),Fraction(5,8)] if multiplier==3 else [Fraction(i,multiplier) for i in range(1,multiplier)]
            expected=[30]
            for start in (30,46):
                expected.extend(start+int(16*f) for f in positions)
                expected.append(start+16)
            self.assertEqual(outputs,expected)
            self.assertEqual(stream.stats['neural_evaluations'],6)
            self.assertEqual(stream.stats['output_frames'],2*multiplier+1)
            self.assertTrue(all(session.closed for session in NativeInterpolation.instances))
            self.assertIsNone(stream.previous)

    def test_scene_cut_repeats_enhanced_endpoints_and_keeps_alpha(self):
        stream=sequence.CombinedEnhancementSequence(dict(self.setup,frame_count=2,interpolation_multiplier=3),JobController())
        stream.open()
        try:
            frames,_=stream.push_frames(temporal.png((10,10,10,100)),0)
            frames,cut=stream.push_frames(temporal.png((240,240,240,220)),1)
            self.assertTrue(cut)
            self.assertEqual([int(f[0,0,0]) for f in frames],[20,250,250])
            self.assertEqual([int(f[0,0,3]) for f in frames],[100,220,220])
        finally: stream.close()

    def test_single_frame_and_disabled_interpolation_create_no_dlssg_session(self):
        for options in (dict(frame_count=1),dict(enable_frame_interpolation=False)):
            setup=dict(self.setup,**options)
            stream=sequence.CombinedEnhancementSequence(setup,JobController())
            stream.open()
            try:
                for i in range(setup['frame_count']):
                    self.assertEqual(len(stream.push_frames(temporal.png((20,20,20,255)),i)[0]),1)
            finally: stream.close()
        self.assertFalse(NativeInterpolation.instances)

    def test_interpolation_initialization_failure_releases_neural_session(self):
        stream=sequence.CombinedEnhancementSequence(self.setup,JobController())
        with patch.object(interpolation,'probe_frame_interpolation_capabilities',return_value=SimpleNamespace(available=False,detail='disabled')):
            with self.assertRaises(RuntimeError): stream.open()
        stream.close(abort=True)
        self.assertTrue(temporal.Neural.instances[-1].closed)

    def test_invalid_protocol_and_multiplier_rejected(self):
        for override in (dict(version=1),dict(interpolation_multiplier=5),dict(enable_frame_interpolation='yes')):
            with self.assertRaises(ValueError): sequence.validate_setup(dict(self.setup,**override))

    def test_real_socket_contract_one_upload_and_multiple_ordered_downloads(self):
        app=Starlette(routes=[WebSocketRoute('/vts/enhance_sequence',sequence.enhancement_socket)])
        with TestClient(app) as client,client.websocket_connect('/vts/enhance_sequence') as socket:
            socket.send_json(dict(self.setup,interpolation_multiplier=4))
            ready=socket.receive_json()
            self.assertEqual((ready['version'],ready['output_count']),(2,9))
            output_index=0
            for index,color in enumerate((20,36,52)):
                data=temporal.png((color,color,color,180))
                socket.send_json(dict(type='frame',index=index,bytes=len(data)))
                socket.send_bytes(data)
                for _ in range(1 if index==0 else 4):
                    header=socket.receive_json()
                    self.assertEqual((header['type'],header['index']),('enhanced',output_index))
                    data=bytearray()
                    while len(data)<header['bytes']: data.extend(socket.receive_bytes())
                    with Image.open(io.BytesIO(data)) as image:
                        self.assertEqual(image.mode,'RGBA')
                        self.assertEqual(image.getpixel((0,0))[3],180)
                    output_index+=1
                self.assertEqual(socket.receive_json(),dict(type='frame_done',index=index))
            socket.send_json(dict(type='end'))
            done=socket.receive_json()
            self.assertEqual(done['type'],'done')
            self.assertEqual(done['stats']['frames'],3)
            self.assertEqual(done['stats']['output_frames'],9)
            self.assertTrue(temporal.Neural.instances[-1].closed)
            self.assertTrue(all(s.closed for s in NativeInterpolation.instances))

    def test_combined_rgb_output_has_no_added_alpha_channel(self):
        stream=sequence.CombinedEnhancementSequence(dict(self.setup,channels=3,frame_count=2),JobController())
        stream.open()
        try:
            for i in range(2):
                frames,_=stream.push_frames(temporal.png((20+i,20+i,20+i),mode='RGB'),i)
                for frame in frames:
                    with Image.open(io.BytesIO(sequence.encode_frame(frame,3))) as image:
                        self.assertEqual(image.mode,'RGB')
        finally: stream.close()


if __name__=='__main__': unittest.main()
