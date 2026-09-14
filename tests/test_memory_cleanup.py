import asyncio
import gc
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import jobs, memory_cleanup
from src.core.neural_worker import Worker
from src.neural_rendering.image import encoder
from PIL import Image


class ManualTimer:
    timers = []
    def __init__(self, seconds, callback):
        self.seconds, self.callback = seconds, callback
        self.cancelled = self.started = False
        self.timers.append(self)
    def start(self): self.started = True
    def cancel(self): self.cancelled = True
    def fire(self): self.callback()  # Can simulate an already-dispatched callback.


class IdleCleanupTests(unittest.TestCase):
    def setUp(self):
        jobs._cancel_idle_timer()
        ManualTimer.timers = []
        for name, value in [('Timer', ManualTimer)]:
            p = patch.object(jobs.threading, name, value)
            p.start(); self.addCleanup(p.stop)
        self.cleanup = Mock()
        self.stop = Mock()
        for name, value in [('_cleanup_job_memory', self.cleanup), ('_stop_idle_worker', self.stop)]:
            p = patch.object(jobs, name, value)
            p.start(); self.addCleanup(p.stop)
        self.addCleanup(jobs._cancel_idle_timer)
        self.assertIsNone(jobs._ACTIVE)
        self.assertFalse(jobs._WAITING)

    def test_timer_starts_only_after_job_and_fires_once(self):
        with jobs.active_job():
            self.assertFalse(ManualTimer.timers)
        self.cleanup.assert_called_once()
        timer = jobs._IDLE_TIMER
        self.assertEqual(timer.seconds, 10)
        timer.fire(); timer.fire()
        self.stop.assert_called_once()
        self.assertIsNone(jobs._IDLE_TIMER)

    def test_new_arrival_cancels_timer_and_stale_callback_cannot_stop_worker(self):
        with jobs.active_job(): pass
        old = jobs._IDLE_TIMER
        with jobs.active_job():
            self.assertTrue(old.cancelled)
            old.fire()
            self.stop.assert_not_called()
            self.assertIsNone(jobs._IDLE_TIMER)
        new = jobs._IDLE_TIMER
        self.assertIsNot(old, new)
        old.fire()
        self.stop.assert_not_called()
        new.fire()
        self.stop.assert_called_once()

    def test_waiting_job_prevents_timer_and_cancelled_last_waiter_starts_it(self):
        waiter = jobs.JobController()
        with jobs.active_job():
            ticket = jobs._enqueue(waiter)
        self.assertIsNone(jobs._IDLE_TIMER)
        waiter.stop()
        jobs._leave(ticket, waiter, False)
        self.assertIsNotNone(jobs._IDLE_TIMER)
        self.cleanup.assert_called_once()  # A cancelled waiter does not clean its owner's memory.

    def test_cleanup_keeps_slot_and_does_not_cancel_next_job(self):
        waiter = jobs.JobController()
        ticket = None
        def clean():
            self.assertIsNotNone(jobs._ACTIVE)
            self.assertFalse(jobs._try_claim(ticket, waiter))
        self.cleanup.side_effect = clean
        with jobs.active_job(): ticket = jobs._enqueue(waiter)
        self.assertTrue(jobs._try_claim(ticket, waiter))
        self.cleanup.side_effect = None
        jobs._leave(ticket, waiter, True)

    def test_failure_and_cancellation_still_cleanup(self):
        for error in (ValueError('bad job'), jobs.Cancelled('stopped')):
            with self.assertRaises(type(error)):
                with jobs.active_job(): raise error
            self.assertIsNone(jobs._ACTIVE)
            self.assertIsNotNone(jobs._IDLE_TIMER)
        self.assertEqual(self.cleanup.call_count, 2)

    def test_cleanup_failure_cannot_strand_queue_or_replace_result(self):
        self.cleanup.side_effect = RuntimeError('cleanup failed')
        with self.assertLogs(jobs.__name__, 'ERROR'):
            with jobs.active_job(): pass
        self.assertIsNone(jobs._ACTIVE)
        self.assertIsNotNone(jobs._IDLE_TIMER)

    def test_enqueue_stays_responsive_but_cannot_claim_during_idle_stop(self):
        entered, release = threading.Event(), threading.Event()
        def stop():
            entered.set()
            self.assertTrue(release.wait(3))
        self.stop.side_effect = stop
        with jobs.active_job(): pass
        timer = jobs._IDLE_TIMER
        controller = jobs.JobController()
        with ThreadPoolExecutor(2) as pool:
            stopping = pool.submit(timer.fire)
            self.assertTrue(entered.wait(3))
            arrival = pool.submit(jobs._enqueue, controller)
            try:
                ticket = arrival.result(3)
                self.assertFalse(jobs._try_claim(ticket, controller))
            finally:
                release.set()
            stopping.result(3)
        self.assertTrue(jobs._try_claim(ticket, controller))
        jobs._leave(ticket, controller, True)

    def test_async_cleanup_does_not_block_event_loop(self):
        entered, release = threading.Event(), threading.Event()
        def clean():
            entered.set()
            if not release.wait(3): raise AssertionError('event loop blocked')
        self.cleanup.side_effect = clean
        class Socket:
            async def receive(self): await asyncio.Future()
        async def check():
            async def run():
                async with jobs.queued_socket_job(Socket(), jobs.JobController(), {}): pass
            task = asyncio.create_task(run())
            try:
                for _ in range(100):
                    if entered.is_set(): break
                    await asyncio.sleep(.01)
                self.assertTrue(entered.is_set())
                self.assertIsNotNone(jobs._ACTIVE)
            finally:
                release.set()
            await task
            self.assertIsNone(jobs._ACTIVE)
        asyncio.run(check())

    def test_failed_idle_shutdown_releases_reservation(self):
        self.stop.side_effect = RuntimeError('shutdown failed')
        with jobs.active_job(): pass
        with self.assertLogs(jobs.__name__, 'ERROR'):
            jobs._IDLE_TIMER.fire()
        self.assertFalse(jobs._STOPPING_WORKER)
        self.assertIsNone(jobs._IDLE_TIMER)
        with jobs.active_job(): pass

    def test_child_queue_does_not_schedule_parent_shutdown(self):
        with patch.dict('os.environ', MERSERK_NEURAL_WORKER='1'):
            with jobs.active_job(): pass
        self.assertIsNone(jobs._IDLE_TIMER)


class MemoryReleaseTests(unittest.TestCase):
    def tearDown(self): encoder._clear_preview_cache()

    def test_preview_pixels_survive_ram_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(encoder, 'GRADIO_TEMP', Path(directory)):
            key = str((Path(directory) / 'output.png').resolve())
            pixels = Image.new('RGBA', (8, 8), (50, 60, 70, 80))
            encoder._preview_cache[key] = (pixels, 256)
            encoder._preview_cache_bytes = 256
            encoder.release_preview_memory()
            self.assertEqual(encoder._preview_cache_bytes, 0)
            self.assertFalse(encoder._preview_cache)
            with encoder.take_image_preview(key) as preview:
                self.assertEqual(preview.getpixel((0, 0)), (50, 60, 70, 80))
            self.assertFalse(list(Path(directory).iterdir()))

    def test_failed_preview_spill_preserves_pending_result(self):
        key = str(Path('not-saved.png').resolve())
        pixels = Image.new('RGB', (8, 8), 'red')
        encoder._preview_cache[key] = (pixels, 192)
        encoder._preview_cache_bytes = 192
        with patch.object(encoder, '_spill_preview'):
            encoder.release_preview_memory()
        self.assertIs(encoder._preview_cache[key][0], pixels)

    def test_local_cleanup_releases_mappings_and_unused_gpu_cache(self):
        prepared, cuda = Mock(), Mock()
        cuda.is_initialized.return_value = True
        modules = {'src.core.runtime': SimpleNamespace(_PREPARED=prepared),
                   'torch': SimpleNamespace(cuda=cuda)}
        with patch.dict(sys.modules, modules), patch.object(gc, 'collect') as collect:
            memory_cleanup.cleanup_local_memory()
        prepared.close.assert_called_once()
        collect.assert_called_once()
        cuda.empty_cache.assert_called_once()

    def test_preview_cleanup_failure_does_not_skip_other_memory_cleanup(self):
        prepared = Mock()
        with patch.dict(sys.modules, {'src.core.runtime': SimpleNamespace(_PREPARED=prepared)}), \
                patch.object(encoder, 'release_preview_memory', side_effect=OSError('disk full')), \
                patch.object(gc, 'collect') as collect:
            with self.assertRaises(OSError): memory_cleanup.cleanup_local_memory()
        prepared.close.assert_called_once()
        collect.assert_called_once()

    def test_local_cleanup_does_not_initialize_cuda(self):
        cuda = Mock()
        cuda.is_initialized.return_value = False
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)):
            memory_cleanup.cleanup_local_memory()
        cuda.empty_cache.assert_not_called()

    def test_worker_cleanup_never_starts_idle_worker(self):
        worker = Worker()
        with patch.object(worker, 'call') as call:
            worker.cleanup()
        call.assert_not_called()

    def test_worker_cleanup_preserves_healthy_process_and_discards_failed_one(self):
        worker = Worker()
        worker.process = SimpleNamespace(poll=lambda: None)
        with patch.object(worker, 'call') as call, patch.object(worker, 'stop') as stop:
            worker.cleanup()
            self.assertEqual(call.call_args.args[0], {'op': 'cleanup'})
            stop.assert_not_called()
            call.side_effect = RuntimeError('worker unresponsive')
            with self.assertLogs('src.core.neural_worker', 'ERROR'): worker.cleanup()
            stop.assert_called_once()


if __name__ == '__main__': unittest.main()
