import asyncio
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import jobs
from src.core.neural_bridge import NeuralBridgeManager, NeuralBridgePoisonedError


class QueueTests(unittest.TestCase):
    def wait_count(self, count):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with jobs._ACTIVE_LOCK:
                if len(jobs._WAITING) == count:
                    return
            time.sleep(.01)
        self.fail('Queue did not reach expected length')

    def test_fifo_and_cancelled_waiter_does_not_touch_owner(self):
        order = []
        controllers = [jobs.JobController() for _ in range(3)]
        def run(index):
            try:
                with jobs.active_job(controllers[index]):
                    order.append(index)
            except jobs.Cancelled:
                return 'cancelled'
        with patch.object(jobs, '_report_wait'), ThreadPoolExecutor(3) as pool:
            owner = jobs.JobController()
            with jobs.active_job(owner):
                futures = []
                for i in range(3):
                    futures.append(pool.submit(run, i)); self.wait_count(i+1)
                controllers[1].stop()
                self.assertEqual(futures[1].result(timeout=3), 'cancelled')
                self.assertFalse(owner.cancel.is_set())
                self.assertEqual(order, [])
            for future in futures: future.result(timeout=3)
        self.assertEqual(order, [0,2])
        self.assertIsNone(jobs._ACTIVE)
        self.assertFalse(jobs._WAITING)

    def test_async_wait_keeps_event_loop_alive_and_cancellation_removes_ticket(self):
        class Socket:
            async def receive(self): await asyncio.Future()
            async def send_json(self, message): pass
        async def check():
            with jobs.active_job():
                async def wait():
                    async with jobs.queued_socket_job(Socket(), jobs.JobController(), {'queue_status':True}):
                        self.fail('Another request owns the slot')
                task = asyncio.create_task(wait())
                await asyncio.sleep(.15)
                self.assertFalse(task.done())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
                self.assertFalse(jobs._WAITING)
        asyncio.run(check())

    def test_failed_owner_releases_slot(self):
        with self.assertRaises(ValueError):
            with jobs.active_job(): raise ValueError('render failed')
        with jobs.active_job(): pass
        self.assertIsNone(jobs._ACTIVE)


class BridgeFailureTests(unittest.TestCase):
    def test_failed_native_recovery_requires_restart_and_preserves_reason(self):
        manager = NeuralBridgeManager()
        detail = 'NR pass 1/2 failed: Bridge reinitialization failed after native crash: device missing'
        with self.assertRaisesRegex(NeuralBridgePoisonedError, 'Restart'):
            manager._check_native_failure(detail)
        with self.assertRaisesRegex(NeuralBridgePoisonedError, 'native crash'):
            manager._guard_poison()

    def test_cached_adapter_with_uninitialized_dll_is_not_used_again(self):
        manager = NeuralBridgeManager()
        manager._initialized_ordinal = 0
        def status(buffer, size):
            buffer.value = b'bridge not initialized'
            return 0
        manager._library = SimpleNamespace(dlss5nr_cuda_status=status)
        with patch.object(manager, '_load'):
            for gpu_mode in (True,False):
                with self.assertRaisesRegex(NeuralBridgePoisonedError, 'Restart'):
                    manager.initialize({'index':0}, require_cuda=gpu_mode)

    def test_ordinary_validation_failure_does_not_poison_bridge(self):
        manager = NeuralBridgeManager()
        manager._check_native_failure('Unsupported image dimensions')
        self.assertFalse(manager._poisoned_reason)


if __name__ == '__main__': unittest.main()
