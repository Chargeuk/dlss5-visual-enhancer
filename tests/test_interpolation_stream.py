from pathlib import Path
import io
import sys
import threading
import unittest
from fractions import Fraction
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from src.frame_interpolation import stream


class FakeGPU:
    instances = []
    fail = False

    def __init__(self, setup, controller):
        self.setup, self.controller = setup, controller
        self.positions = [Fraction(i, setup['multiplier']) for i in range(1, setup['multiplier'])]
        self.closed = threading.Event()
        self.instances.append(self)

    def open(self):
        pass

    def push(self, data, index):
        if self.fail:
            raise RuntimeError('GPU test failure')
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (64, 32)
        return [(dict(type='generated', slot=slot, bytes=len(data)), bytes(data))
                for slot in range(1, self.setup['multiplier']) if index], False

    def close(self):
        self.closed.set()


class ServerTests(unittest.TestCase):
    def setUp(self):
        FakeGPU.instances, FakeGPU.fail = [], False
        self.patch = patch.object(stream, 'InterpolationStream', FakeGPU)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = TestClient(Starlette(routes=[WebSocketRoute('/vts/interpolate', stream.interpolation_socket)]))
        self.setup = dict(version=1, width=64, height=32, frame_count=2, multiplier=3)
        with Image.new('RGBA', (64, 32), 'red') as image, io.BytesIO() as buffer:
            image.save(buffer, format='PNG')
            self.png = buffer.getvalue()

    def connect(self):
        return self.client.websocket_connect('/vts/interpolate')

    def frame(self, socket, index):
        socket.send_json(dict(type='frame', index=index, bytes=len(self.png)))
        socket.send_bytes(self.png[:10])
        socket.send_bytes(self.png[10:])

    def test_chunks_and_only_generated_outputs_then_slot_is_reusable(self):
        for _ in range(2):
            with self.connect() as socket:
                socket.send_json(self.setup)
                self.assertEqual(socket.receive_json()['output_count'], 4)
                self.frame(socket, 0)
                self.frame(socket, 1)
                socket.send_json(dict(type='end'))
                self.assertEqual(socket.receive_json()['type'], 'frame_done')
                for slot in (1, 2):
                    header = socket.receive_json()
                    self.assertEqual((header['type'], header['slot']), ('generated', slot))
                    self.assertEqual(socket.receive_bytes(), self.png)
                self.assertEqual(socket.receive_json()['type'], 'frame_done')
                self.assertEqual(socket.receive_json()['type'], 'done')
                self.assertTrue(FakeGPU.instances[-1].closed.is_set())

    def test_disconnect_releases_gpu(self):
        with self.connect() as socket:
            socket.send_json(self.setup)
            socket.receive_json()
        self.assertTrue(FakeGPU.instances[-1].closed.wait(3))
        with self.connect() as socket:
            socket.send_json(self.setup)
            self.assertEqual(socket.receive_json()['type'], 'ready')

    def test_bad_sequence_and_gpu_failure_cleanup(self):
        for fail in (False, True):
            FakeGPU.fail = fail
            with self.connect() as socket:
                socket.send_json(self.setup)
                socket.receive_json()
                self.frame(socket, 0 if fail else 7)
                self.assertEqual(socket.receive_json()['type'], 'error')
                self.assertTrue(FakeGPU.instances[-1].closed.is_set())

    def test_concurrent_request_does_not_cancel_current_owner(self):
        with self.connect() as first:
            first.send_json(self.setup)
            first.receive_json()
            with self.connect() as second:
                second.send_json(dict(self.setup, queue_status=True))
                self.assertEqual(second.receive_json(), dict(type='queued', position=1))
                self.assertEqual(len(FakeGPU.instances), 1)
            self.assertFalse(FakeGPU.instances[0].controller.cancel.is_set())

    def test_invalid_setup(self):
        with self.connect() as socket:
            socket.send_json(dict(self.setup, multiplier=5))
            self.assertEqual(socket.receive_json()['type'], 'error')
        self.assertEqual(FakeGPU.instances, [])


if __name__ == '__main__':
    unittest.main()
