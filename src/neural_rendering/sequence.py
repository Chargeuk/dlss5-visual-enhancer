"""Lossless, bounded frame transport around upstream temporal rendering."""
from __future__ import annotations

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import anyio
import numpy as np
from PIL import Image
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..core.gpu_selection import resolve_runtime_ai_gpu
from ..core.jobs import Cancelled, JobController, queued_socket_job
from ..core.runtime import DLSSFrameSession, prepare_runtime, resolve_native_settings, resolve_upscaling_mode, verify_feature_18
from ..core.neural_worker import WorkerLost, RECOVERY_CODE
from ..settings.storage import processing_gpu_settings
from ..upscale.image.processor import srgb_to_worker, worker_to_srgb
from ..upscale.video.models import UpscaleOptions
from ..upscale.video.native import RTXVideoSession, probe_capabilities
from .video.guides import TemporalGuideGenerator
from .video.models import ConversionOptions

CHUNK_BYTES = 256 * 1024
IDLE_TIMEOUT = 120
NEURAL_PARAMETERS = {
    "nr_passes", "nr_style", "nr_intensity", "local_tone_strength",
    "local_structure_strength", "skin_structure_strength", "automatic_mask",
    "nr_color_strength", "tone_preservation", "face_skin_protection",
    "grain_preservation", "shimmer_suppression",
}


def validate_setup(value):
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("VTS temporal enhancement protocol version 1 is required.")
    allowed = {"version", "width", "height", "target_width", "target_height", "frame_count",
               "channels", "enable_neural_rendering", "vsr_quality", "parameters", "timeout_seconds", "queue_status"}
    if value.keys() - allowed:
        raise ValueError("Unknown sequence settings.")
    if not isinstance(value.get("queue_status", False), bool):
        raise ValueError("queue_status must be a boolean.")
    for name in ("width", "height", "target_width", "target_height"):
        n = value.get(name)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 16384:
            raise ValueError(f"{name} must be an integer from 1 to 16384.")
    for names in (("width", "height"), ("target_width", "target_height")):
        if value[names[0]] * value[names[1]] > 100_000_000:
            raise ValueError("Frames must not exceed 100 megapixels.")
    count = value.get("frame_count")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 0xFFFFFFFF:
        raise ValueError("frame_count must be a positive uint32.")
    if value.get("channels") not in (3, 4):
        raise ValueError("Frames must have 3 RGB or 4 RGBA channels.")
    if not isinstance(value.get("enable_neural_rendering", True), bool):
        raise ValueError("enable_neural_rendering must be a boolean.")
    timeout = value.get("timeout_seconds", 600)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 86400:
        raise ValueError("timeout_seconds must be between 1 and 86400.")
    quality = value.get("vsr_quality", 4)
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 4:
        raise ValueError("vsr_quality must be between 1 and 4.")
    params = value.get("parameters", {})
    if not isinstance(params, dict) or params.keys() - NEURAL_PARAMETERS:
        raise ValueError("Unknown neural controls; outer iterations and HDR are not supported.")
    if value.get("enable_neural_rendering", True):
        sizes = (value["target_width"], value["target_height"])
        if min(sizes) < 64 or max(sizes) > 7680 or min(sizes) > 4320:
            raise ValueError("Neural output must be at least 64 per side and fit within 7680 by 4320 (either orientation).")
        resolve_native_settings(ConversionOptions(**params))
    elif params:
        raise ValueError("Neural controls require neural rendering to be enabled.")
    return value


class EnhancementSequence:
    """One persistent VSR session and one neural history per ordered sequence."""

    def __init__(self, setup, controller):
        self.setup, self.controller = setup, controller
        self.size = (setup["width"], setup["height"])
        self.target = (setup["target_width"], setup["target_height"])
        self.reduced = tuple(min(a, b) for a, b in zip(self.size, self.target))
        self.neural = self.vsr = self.guides = None
        self.next_index = 0
        self.scene_cuts = 0
        self.stats = {}

    def open(self):
        gpu_uuid = processing_gpu_settings()[0]
        if self.reduced != self.target:
            caps = probe_capabilities(gpu_uuid, controller=self.controller)
            self.vsr = RTXVideoSession(*self.reduced, *self.target,
                UpscaleOptions(ai_gpu_uuid=gpu_uuid, vsr_quality=self.setup.get("vsr_quality", 4)),
                1, caps, self.controller, timeout=self.setup.get("timeout_seconds", 600))
        if self.setup.get("enable_neural_rendering", True):
            prepared = prepare_runtime()
            gpu = resolve_runtime_ai_gpu(prepared.gpus, prepared.runtime_bundle, gpu_uuid)
            options = ConversionOptions(**self.setup.get("parameters", {}))
            factor, mode = resolve_upscaling_mode(1.0)
            self.neural = DLSSFrameSession(input_width=self.target[0], input_height=self.target[1],
                output_width=self.target[0], output_height=self.target[1],
                frame_count=self.setup["frame_count"], warmup_frames=0, factor=factor, mode=mode,
                native_settings=resolve_native_settings(options), gpu=gpu,
                runtime_bundle=prepared.runtime_bundle, controller=self.controller,
                recover_reset_frames=False)
            # Detect cuts from source frames, before model-created changes.
            self.guides = TemporalGuideGenerator(*self.reduced)

    def push(self, encoded, index):
        if self.controller.cancel.is_set():
            raise Cancelled("Sequence stopped.")
        if index != self.next_index:
            raise ValueError("Input frames must arrive once, in sequence.")
        mode = "RGBA" if self.setup["channels"] == 4 else "RGB"
        with Image.open(io.BytesIO(encoded)) as source:
            if source.format != "PNG" or source.size != self.size or source.mode != mode:
                raise ValueError("Expected an 8-bit PNG with the negotiated dimensions and channels.")
            with source.convert("RGBA") as rgba_image:
                if self.reduced != self.size:
                    with rgba_image.resize(self.reduced, Image.Resampling.LANCZOS) as reduced:
                        rgba = np.array(reduced, dtype=np.uint8)
                else:
                    rgba = np.array(rgba_image, dtype=np.uint8)
        guide = self.guides.process(rgba) if self.guides is not None else None
        if self.vsr is not None:
            alpha = rgba[..., 3].copy() if self.setup["channels"] == 4 else None
            rgba = worker_to_srgb(self.vsr.process_frame(srgb_to_worker(rgba)), *self.target, alpha)
        if self.neural is not None:
            rgba, _ = self.neural.process(index=index, rgba=rgba, reset=guide.reset, pts=index)
            if index == 0:
                verify_feature_18(self.neural.bridge_logs, self.neural.structured_status())
        cut = bool(index and guide is not None and guide.reset)
        self.scene_cuts += int(cut)
        self.next_index += 1
        if self.controller.cancel.is_set():
            raise Cancelled("Sequence stopped.")
        with Image.fromarray(rgba if mode == "RGBA" else rgba[..., :3]) as result, io.BytesIO() as buffer:
            result.save(buffer, format="PNG", compress_level=1)
            png = buffer.getvalue()
        return png, cut

    def close(self, *, abort=False):
        try:
            if self.neural is not None:
                # Diagnostics must never prevent releasing native resources.
                with suppress(Exception):
                    self.stats = {
                        "neural_evaluations": self.neural.diagnostics.feature_evaluations,
                        "temporal": self.neural.structured_status().get("temporal_stabilization", {}),
                    }
                if abort:
                    self.neural.abort()
                else:
                    self.neural.close()
        finally:
            try:
                if self.vsr is not None:
                    self.vsr.close(abort=abort)
            finally:
                self.stats.update(frames=self.next_index, scene_cuts=self.scene_cuts,
                                  vsr_frames=self.vsr.completed_frames if self.vsr else 0)
                self.neural = self.vsr = self.guides = None


async def enhancement_socket(websocket: WebSocket):
    await websocket.accept()
    controller = JobController()
    receiver = processing = stream = executor = None
    complete = False
    try:
        setup = validate_setup(await asyncio.wait_for(websocket.receive_json(), IDLE_TIMEOUT))
        timeout = setup.get("timeout_seconds", 600)
        incoming = asyncio.Queue(maxsize=1)
        max_png_bytes = setup["width"] * setup["height"] * 8 + 1024 * 1024

        async def receive_frames():
            try:
                for index in range(setup["frame_count"]):
                    header = await asyncio.wait_for(websocket.receive_json(), timeout)
                    if not isinstance(header, dict) or header.get("type") != "frame" or header.get("index") != index:
                        raise ValueError("Expected the next numbered frame.")
                    size = header.get("bytes")
                    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= max_png_bytes:
                        raise ValueError("Invalid PNG payload length.")
                    data = bytearray()
                    while len(data) < size:
                        chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout)
                        if not chunk or len(chunk) > CHUNK_BYTES or len(data) + len(chunk) > size:
                            raise ValueError("Invalid PNG chunk length.")
                        data.extend(chunk)
                    await incoming.put((index, data))
                if await asyncio.wait_for(websocket.receive_json(), timeout) != {"type": "end"}:
                    raise ValueError("Expected the end of the frame sequence.")
                await incoming.put(None)
            except BaseException:
                controller.stop()
                raise

        async def next_input():
            waiting = asyncio.create_task(incoming.get())
            try:
                done, _ = await asyncio.wait((waiting, receiver), return_when=asyncio.FIRST_COMPLETED)
                if receiver in done and not receiver.cancelled() and receiver.exception() is not None:
                    raise receiver.exception()
                return await waiting
            finally:
                if not waiting.done():
                    waiting.cancel()

        async def run(function, *args):
            return await asyncio.get_running_loop().run_in_executor(executor, function, *args)

        async with queued_socket_job(websocket, controller, setup):
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vts-temporal")
            stream = EnhancementSequence(setup, controller)
            receiver = asyncio.create_task(receive_frames())
            try:
                processing = asyncio.create_task(run(stream.open))
                await asyncio.shield(processing)
                await websocket.send_json(dict(type="ready", version=1, chunk_bytes=CHUNK_BYTES,
                    output_count=setup["frame_count"], width=setup["target_width"], height=setup["target_height"],
                    channels=setup["channels"]))
                for index in range(setup["frame_count"]):
                    actual_index, png = await next_input()
                    processing = asyncio.create_task(run(stream.push, png, actual_index))
                    output, cut = await asyncio.shield(processing)
                    del png
                    await asyncio.wait_for(websocket.send_json(dict(type="enhanced", index=index,
                        bytes=len(output), scene_cut=cut)), timeout)
                    for offset in range(0, len(output), CHUNK_BYTES):
                        await asyncio.wait_for(websocket.send_bytes(output[offset:offset + CHUNK_BYTES]), timeout)
                    del output
                if await next_input() is not None:
                    raise ValueError("Unexpected frames after the end of the sequence.")
                complete = True
            finally:
                # Keep the GPU slot until outstanding work and native cleanup finish.
                with anyio.CancelScope(shield=True):
                    if not receiver.done():
                        receiver.cancel()
                    if not complete:
                        controller.stop()
                    if processing is not None and not processing.done():
                        with suppress(Exception):
                            await asyncio.shield(processing)
                    await run(lambda: stream.close(abort=not complete or controller.cancel.is_set()))
        await websocket.send_json(dict(type="done", stats=stream.stats))
    except (WebSocketDisconnect, asyncio.CancelledError):
        controller.stop()
    except Exception as exc:
        with suppress(Exception):
            await websocket.send_json(dict(type="error", message=str(exc),
                code=RECOVERY_CODE if isinstance(exc, WorkerLost) else None))
    finally:
        if receiver is not None:
            receiver.cancel()
            with suppress(BaseException):
                await receiver
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with suppress(Exception):
            await websocket.close()
