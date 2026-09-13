"""Lossless frame streaming for VTS; no video or image files are created."""
from __future__ import annotations

import asyncio
import anyio
import io
import json
from contextlib import suppress
from fractions import Fraction

import numpy as np
from PIL import Image
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..core.jobs import JobController, active_job
from .capabilities import probe_frame_interpolation_capabilities
from .native import DirectDLSSGSession
from .processor import DLSSGStage, TimedFrame

CHUNK_BYTES = 256 * 1024
IDLE_TIMEOUT = 120


def validate_setup(value):
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("VTS interpolation protocol version 1 is required.")
    for name in ("width", "height", "frame_count", "multiplier"):
        if isinstance(value.get(name), bool) or not isinstance(value.get(name), int):
            raise ValueError(f"{name} must be an integer.")
    if value["multiplier"] not in (2, 3, 4, 8):
        raise ValueError("Choose a 2x, 3x, 4x or 8x multiplier.")
    if value["frame_count"] < 2:
        raise ValueError("Interpolation requires at least two input frames.")
    if min(value["width"], value["height"]) < 1 or max(value["width"], value["height"]) > 16384:
        raise ValueError("Frame dimensions must be between 1 and 16384 pixels.")
    if value["width"] * value["height"] > 100_000_000:
        raise ValueError("Frame exceeds the image decoder's 100-megapixel limit.")
    timeout = value.get("timeout_seconds", 600)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("Timeout must be a positive integer number of seconds.")
    return value


class InterpolationStream:
    def __init__(self, setup, controller):
        self.width, self.height = setup["width"], setup["height"]
        self.count, self.multiplier = setup["frame_count"], setup["multiplier"]
        self.controller = controller
        self.grid = 8 if self.multiplier == 3 else self.multiplier
        self.positions = [Fraction(i, self.grid) for i in ((3, 5) if self.multiplier == 3 else range(1, self.grid))]
        self.sessions = []
        self.stages = []
        self.previous_alpha = None
        self.next_index = 0
        self.segment = 0

    def open(self):
        capabilities = probe_frame_interpolation_capabilities()
        if not capabilities.available:
            raise RuntimeError("DLSS Frame Generation is unavailable: " + capabilities.detail)
        try:
            # A persistent 2x cascade has the same timing on every supported GPU.
            for index in range(self.grid.bit_length() - 1):
                expected = (self.count - 1) * (1 << index) + 1
                session = DirectDLSSGSession(self.width, self.height, expected, 1, self.controller)
                self.sessions.append(session)
                self.stages.append(DLSSGStage(session, self.width, self.height, 1,
                                             detect_source_cuts=index == 0))
        except BaseException:
            self.close()
            raise

    def push(self, encoded, index):
        if index != self.next_index:
            raise ValueError("Input frames must arrive once, in sequence.")
        with Image.open(io.BytesIO(encoded)) as source:
            if source.format != "PNG" or source.size != (self.width, self.height):
                raise ValueError("Every input must be a PNG with the negotiated dimensions.")
            rgba = np.array(source.convert("RGBA"), dtype=np.uint8)
        alpha = rgba[..., 3].copy()
        items = [TimedFrame(rgba, Fraction(index, 30), self.segment, "Source", index)]
        cuts_before = self.stages[0].scene_cuts
        for stage in self.stages:
            expanded = []
            for item in items:
                expanded.extend(stage.push(item))
            items = expanded
        cut = self.stages[0].scene_cuts != cuts_before
        self.segment = self.stages[0].previous.segment
        generated = {item.timestamp: item.rgba for item in items if item.provenance == "DLSSG"}
        outputs = []
        if index:
            for slot, fraction in enumerate(self.positions, 1):
                timestamp = (Fraction(index - 1) + fraction) / 30
                if cut:
                    # Keep exactly the same number of output slots across scene cuts.
                    outputs.append((dict(type="repeat", slot=slot,
                                         source_index=index - 1 if fraction <= Fraction(1, 2) else index), None))
                    continue
                if timestamp not in generated:
                    raise RuntimeError("DLSS did not return an expected intermediate frame.")
                pixels = generated[timestamp]
                pixels[..., 3] = np.rint(self.previous_alpha.astype(np.float32) * (1 - float(fraction))
                                        + alpha.astype(np.float32) * float(fraction)).astype(np.uint8)
                with Image.fromarray(pixels) as image, io.BytesIO() as buffer:
                    image.save(buffer, format="PNG", compress_level=1)
                    data = buffer.getvalue()
                outputs.append((dict(type="generated", slot=slot, bytes=len(data)), data))
        self.previous_alpha = alpha
        self.next_index += 1
        return outputs, cut

    def close(self):
        for session in reversed(self.sessions):
            session.close()
        self.sessions.clear()
        self.stages.clear()
        self.previous_alpha = None


async def interpolation_socket(websocket: WebSocket):
    await websocket.accept()
    controller = JobController()
    receiver = None
    stream = None
    processing = None
    try:
        setup = validate_setup(await asyncio.wait_for(websocket.receive_json(), IDLE_TIMEOUT))
        incoming = asyncio.Queue(maxsize=1)
        max_png_bytes = setup["width"] * setup["height"] * 8 + 1024 * 1024
        input_timeout = setup.get("timeout_seconds", 600)

        async def receive_frames():
            try:
                for index in range(setup["frame_count"]):
                    header = await asyncio.wait_for(websocket.receive_json(), input_timeout)
                    if header.get("type") != "frame" or header.get("index") != index:
                        raise ValueError("Expected the next numbered frame.")
                    size = header.get("bytes")
                    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= max_png_bytes:
                        raise ValueError("Invalid PNG payload length.")
                    data = bytearray()
                    while len(data) < size:
                        chunk = await asyncio.wait_for(websocket.receive_bytes(), input_timeout)
                        if not chunk or len(chunk) > CHUNK_BYTES or len(data) + len(chunk) > size:
                            raise ValueError("Invalid PNG chunk length.")
                        data.extend(chunk)
                    await incoming.put((index, data))
                end = await asyncio.wait_for(websocket.receive_json(), input_timeout)
                if end != {"type": "end"}:
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

        with active_job(controller):
            stream = InterpolationStream(setup, controller)
            receiver = asyncio.create_task(receive_frames())
            try:
                processing = asyncio.create_task(asyncio.to_thread(stream.open))
                await asyncio.shield(processing)
                await websocket.send_json(dict(type="ready", version=1, chunk_bytes=CHUNK_BYTES,
                    output_count=(setup["frame_count"] - 1) * setup["multiplier"] + 1,
                    positions=[[p.numerator, p.denominator] for p in stream.positions]))
                for index in range(setup["frame_count"]):
                    actual_index, png = await next_input()
                    processing = asyncio.create_task(asyncio.to_thread(stream.push, png, actual_index))
                    outputs, cut = await asyncio.shield(processing)
                    del png
                    for header, data in outputs:
                        await asyncio.wait_for(websocket.send_json(header), IDLE_TIMEOUT)
                        if data is not None:
                            for offset in range(0, len(data), CHUNK_BYTES):
                                await asyncio.wait_for(websocket.send_bytes(data[offset:offset + CHUNK_BYTES]), IDLE_TIMEOUT)
                    del outputs
                    await websocket.send_json(dict(type="frame_done", index=index, scene_cut=cut))
                if await next_input() is not None:
                    raise ValueError("Unexpected frames after the end of the sequence.")
            finally:
                # ASGI disconnect/shutdown cancellation must not skip native cleanup.
                # Shield in-flight worker tasks too: cancelling a to_thread task does
                # not stop its underlying thread, so wait before closing its session.
                with anyio.CancelScope(shield=True):
                    if receiver is not None and not receiver.done():
                        receiver.cancel()
                    controller.stop()
                    if processing is not None and not processing.done():
                        with suppress(Exception):
                            await asyncio.shield(processing)
                    await asyncio.to_thread(stream.close)
        await websocket.send_json(dict(type="done"))
    except (WebSocketDisconnect, asyncio.CancelledError):
        controller.stop()
    except Exception as exc:
        with suppress(Exception):
            await websocket.send_json(dict(type="error", message=str(exc)))
    finally:
        if receiver is not None:
            receiver.cancel()
            with suppress(BaseException):
                await receiver
        with suppress(Exception):
            await websocket.close()
