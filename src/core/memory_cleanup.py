"""Release disposable job memory without invalidating pending results."""
from __future__ import annotations

import gc
import os
import sys


def cleanup_local_memory():
    # Do not import/initialize a GPU framework just to clear its allocator.
    try:
        encoder = sys.modules.get('src.neural_rendering.image.encoder')
        if encoder is not None:
            encoder.release_preview_memory()
    finally:
        try:
            runtime = sys.modules.get('src.core.runtime')
            prepared = getattr(runtime, '_PREPARED', None)
            if prepared is not None:
                # Keep small device/settings metadata, release warmed file mappings.
                prepared.close()
        finally:
            gc.collect()
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def cleanup_job_memory():
    try:
        if os.environ.get('MERSERK_NEURAL_WORKER') != '1':
            worker_module = sys.modules.get('src.core.neural_worker')
            if worker_module is not None:
                worker_module.WORKER.cleanup()
    finally:
        cleanup_local_memory()


def stop_idle_worker():
    worker_module = sys.modules.get('src.core.neural_worker')
    if worker_module is not None:
        worker_module.WORKER.stop()
    cleanup_local_memory()
