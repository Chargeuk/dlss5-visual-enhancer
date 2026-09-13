from __future__ import annotations

import os
import time
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image

from ...core.batch_progress import BatchItemUpdate, BatchProgress
from ...core.disk_paths import OutputFile, prepare_output_dir
from ...core.gpu_selection import resolve_runtime_ai_gpu
from ...core.iterations import validate_iterations
from ...core.jobs import Cancelled, JobController, active_job
from ...core.render_metadata import prepare_render_note
from ...core.naming import output_filename, validate_rename
from ...core.paths import LOGS, OUTPUTS
from ...core.runtime import (
    DLSSFrameSession, prepare_runtime, resize_fit, resolve_native_settings,
    resolve_output_size, resolve_upscaling_mode, verify_feature_18, write_failure_report,
)
from .decoder import decode_image
from .encoder import _encode_image, take_image_preview
from .models import NO_SAVE, IMAGE_EXTENSIONS, IMAGE_FORMATS, ImageBatchResult, ImageConversionFailure, ImageConversionOptions, ImageConversionResult
from .reports import _build_manifest_and_zip, _report_data, _write_report


def _validate_options(options: ImageConversionOptions) -> ImageConversionOptions:
    options.iterations = validate_iterations(options.iterations)
    if options.target_width is not None or options.target_height is not None:
        sizes = (options.target_width, options.target_height)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 64 for n in sizes):
            raise ValueError("DLSS target width and height must both be integers of at least 64 pixels.")
        if max(sizes) > 7680 or min(sizes) > 4320:
            raise ValueError("DLSS targets support a longest side up to 7680 and a shortest side up to 4320 pixels.")
    if options.output_format not in IMAGE_FORMATS:
        raise ValueError(f"Unknown image output format: {options.output_format!r}.")
    if isinstance(options.quality, bool):
        raise ValueError("Image quality must be an integer from 1 to 100.")
    try:
        quality = int(options.quality)
    except (TypeError, ValueError) as exc:
        raise ValueError("Image quality must be an integer from 1 to 100.") from exc
    if quality != options.quality or not 1 <= quality <= 100:
        raise ValueError("Image quality must be an integer from 1 to 100.")
    validate_rename(options.rename_mode, options.custom_suffix)
    resolve_native_settings(options)
    resolve_upscaling_mode(options.upscaling_factor)
    return options


def _output_path(source, output_format, stamp, index, rename_mode, custom_suffix, *, output_dir=None):
    safe_stem = source.stem.strip().rstrip(".") or "image"
    return (output_dir or OUTPUTS) / output_filename(
        source, IMAGE_EXTENSIONS[output_format], rename_mode, custom_suffix,
        f"{safe_stem}_DLSS5_IMAGE_{stamp}-{index + 1:04d}",
    )


def convert_images(
    input_paths: Iterable[str | os.PathLike[str]],
    options: ImageConversionOptions | None = None,
    progress: Callable[[float, str], None] | None = None,
    *, output_dir: str | os.PathLike[str] | None = None,
    controller: JobController | None = None,
    on_item_update: Callable[[BatchItemUpdate], None] | None = None,
    generate_previews: bool = True, create_zip: bool = True,
    on_image: Callable[[Image.Image, str], None] | None = None,
) -> ImageBatchResult:
    options = _validate_options(replace(options) if options else ImageConversionOptions())
    inputs = list(input_paths)
    paths = [Path(f"memory-image-{i + 1}.png") if isinstance(path, Image.Image) else Path(path).resolve()
             for i, path in enumerate(inputs)]
    save_output = options.output_format != NO_SAVE
    if not paths:
        raise ValueError("Choose at least one image.")
    controller = controller or JobController()
    reporter = BatchProgress(paths, on_item_update, progress)
    successes, failures = [], []
    session = None
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    try:
        destination = prepare_output_dir(output_dir, default=OUTPUTS) if save_output else None
        LOGS.mkdir(exist_ok=True)
        with active_job(controller), ThreadPoolExecutor(max_workers=1, thread_name_prefix="dlss5-image-decode") as decoder:
            if controller.cancel.is_set():
                raise Cancelled("Stopped before rendering.")
            prepared = prepare_runtime()
            gpu = resolve_runtime_ai_gpu(prepared.gpus, prepared.runtime_bundle, options.ai_gpu_uuid)
            factor, mode = resolve_upscaling_mode(options.upscaling_factor)
            native = resolve_native_settings(options)
            next_decode = None
            session_size = None
            sent = 0
            try:
                for index, path in enumerate(paths):
                    if controller.cancel.is_set():
                        break
                    reporter.advance(index, 0.0, "Decoding image")
                    output_file = None
                    output = (_output_path(path, options.output_format, stamp, index,
                                           options.rename_mode, options.custom_suffix, output_dir=destination)
                              if save_output else None)
                    decoded = processed = render_rgba = current_rgba = alpha = None
                    try:
                        future = next_decode or decoder.submit(decode_image, inputs[index])
                        next_decode = None
                        while True:
                            if controller.cancel.is_set():
                                future.cancel()
                                raise Cancelled("Image rendering stopped by user.")
                            try:
                                decoded = future.result(timeout=.1)
                                break
                            except TimeoutError:
                                continue
                        del future
                        # Only one future image may be decoded ahead of the active file.
                        if index + 1 < len(paths):
                            next_decode = decoder.submit(decode_image, inputs[index + 1])
                        height, width = decoded.rgba.shape[:2]
                        if width < 64 or height < 64:
                            raise ValueError(f"{path.name} is {width}Ã—{height}; DLSS requires at least 64Ã—64.")
                        output_width, output_height = (
                            (options.target_width, options.target_height) if options.target_width is not None
                            else resolve_output_size(width, height, options.upscaling_factor)
                        )
                        output_file = OutputFile(output) if save_output else None
                        dimensions = (output_width, output_height)
                        alpha = (decoded.alpha if decoded.alpha.shape == dimensions[::-1] else
                                 cv2.resize(decoded.alpha, dimensions, interpolation=cv2.INTER_LANCZOS4))
                        current_rgba = decoded.rgba
                        iteration_reports = []
                        for iteration in range(options.iterations):
                            if controller.cancel.is_set():
                                raise Cancelled("Image rendering stopped by user.")
                            pass_factor, pass_mode = (factor, mode) if iteration == 0 else resolve_upscaling_mode(1.0)
                            pass_height, pass_width = current_rgba.shape[:2]
                            # A native-resolution enhancement needs a different session from upscaling.
                            session_key = (pass_width, pass_height, output_width, output_height, pass_factor)
                            label = f"Iteration {iteration + 1}/{options.iterations}"
                            fraction = .1 + .65 * iteration / options.iterations
                            if session is not None and session_key != session_size:
                                session.close()
                                session = None
                            if session is None:
                                reporter.advance(index, fraction, f"{label}: starting DLSS")
                                session = DLSSFrameSession(
                                    input_width=pass_width, input_height=pass_height,
                                    output_width=output_width, output_height=output_height,
                                    frame_count=None, warmup_frames=options.warmup_frames,
                                    factor=pass_factor, mode=pass_mode, native_settings=native, gpu=gpu,
                                    runtime_bundle=prepared.runtime_bundle, controller=controller,
                                )
                                session_size, sent = session_key, 0
                                motion = np.zeros((session.render_height, session.render_width, 2), dtype=np.float16)
                            action = "upscaling and enhancing" if iteration == 0 and factor != 1.0 else "enhancing"
                            reporter.advance(index, fraction, f"{label}: {action}")
                            # Exact-size API callers already choose crop or stretch; do not add letterboxing.
                            render_rgba = (
                                cv2.resize(current_rgba, (session.render_width, session.render_height),
                                           interpolation=cv2.INTER_LANCZOS4)
                                if options.target_width is not None
                                else resize_fit(current_rgba, session.render_width, session.render_height)
                            )
                            processed, _pts = session.process(index=sent, rgba=render_rgba, motion=motion, reset=True, pts=sent)
                            sent += 1
                            evidence = verify_feature_18(session.worker_logs, session.reshade_log_text())
                            processed[..., 3] = alpha
                            iteration_reports.append({
                                "iteration": iteration + 1,
                                "input_dimensions": {"width": pass_width, "height": pass_height},
                                "render_dimensions": {"width": session.render_width, "height": session.render_height},
                                "output_dimensions": {"width": output_width, "height": output_height},
                                "upscaling_factor": pass_factor, "dlss_mode": pass_mode["name"],
                                "feature_18_confirmed": True, "history_reset": True,
                                "dlssnr_evidence": evidence["evidence"],
                            })
                            # Feed pixels directly into the next pass; encoding happens only after the loop.
                            current_rgba = processed
                            render_rgba = None
                            reporter.advance(index, .1 + .65 * (iteration + 1) / options.iterations, f"{label}: complete")
                        if controller.cancel.is_set():
                            raise Cancelled("Image rendering stopped by user.")
                        if not save_output:
                            if on_image is not None:
                                with Image.fromarray(processed) as rendered:
                                    on_image(rendered, path.name)
                            if controller.cancel.is_set():
                                raise Cancelled("Image rendering stopped by user.")
                            result = ImageConversionResult(
                                str(path), "", "", reporter.elapsed(index), str(gpu["display_name"]),
                                width, height, session.render_width, session.render_height, output_width, output_height,
                                float(options.upscaling_factor), str(session.mode["name"]), options.output_format,
                                options.dlss_model_preset, session.applied_dlss_model_preset, list(decoded.warnings),
                            )
                            successes.append(result)
                            reporter.complete(index, "", "Not saved; neural rendering verified.")
                            continue
                        reporter.advance(index, .8, "Saving image")
                        metadata_diagnostics = {}
                        render_note = prepare_render_note(options, session.applied_dlss_model_preset,
                                                          metadata_diagnostics)
                        warnings = _encode_image(output_file.temporary, processed, options, decoded.metadata,
                                                 generate_preview=generate_previews, preview_path=output,
                                                 render_note=render_note, metadata_diagnostics=metadata_diagnostics,
                                                 controller=controller)
                        if metadata_diagnostics.get("warning") not in warnings and metadata_diagnostics.get("warning"):
                            warnings.append(metadata_diagnostics["warning"])
                        if controller.cancel.is_set():
                            raise Cancelled("Image rendering stopped by user.")
                        reporter.advance(index, .95, "Verifying image and writing report")
                        if index == len(paths) - 1:
                            session.close()
                        result = ImageConversionResult(
                            str(path), str(output), "", reporter.elapsed(index), str(gpu["display_name"]),
                            width, height, session.render_width, session.render_height, output_width, output_height,
                            float(options.upscaling_factor), str(session.mode["name"]), options.output_format,
                            options.dlss_model_preset, session.applied_dlss_model_preset, [*decoded.warnings, *warnings],
                        )
                        result.report_path = _write_report(result, options, _report_data(decoded), gpu, session, evidence,
                                                          metadata_diagnostics=metadata_diagnostics, iteration_reports=iteration_reports)
                        if controller.cancel.is_set():
                            raise Cancelled("Image rendering stopped by user.")
                        output_file.publish()
                        successes.append(result)
                        reporter.complete(index, str(output), "; ".join([*result.warnings, f"Report: {result.report_path}"]))
                    except Exception as exc:
                        cancelled = isinstance(exc, Cancelled) or controller.cancel.is_set()
                        if output_file is not None:
                            output_file.cleanup(rollback=True)
                        preview = take_image_preview(output) if save_output else None
                        if preview is not None:
                            preview.close()
                        if session is not None:
                            session.abort()
                            if not cancelled:
                                with suppress(Exception):
                                    report = write_failure_report(
                                        operation="image-render", source=str(path), error=exc, gpu=gpu,
                                        runtime_bundle=prepared.runtime_bundle, worker_logs=session.worker_logs,
                                        reshade_lines=session.reshade_log_text().splitlines(), logs_dir=LOGS,
                                    )
                                    exc = RuntimeError(f"{exc}\nDiagnostic report: {report}")
                            session = None
                        failures.append(ImageConversionFailure(str(path), str(exc)))
                        reporter.fail(index, exc, cancelled=cancelled)
                        if cancelled:
                            controller.stop()
                            break
                    finally:
                        if output_file is not None:
                            output_file.cleanup()
                        decoded = processed = render_rgba = current_rgba = alpha = None
                if session is not None and not session.closed:
                    if controller.cancel.is_set():
                        session.abort()
                    else:
                        session.close()
            finally:
                if next_decode is not None:
                    next_decode.cancel()
                if session is not None and not session.closed:
                    session.abort()
        cancelled = controller.cancel.is_set()
        if cancelled:
            for item in reporter.items:
                if item.state == "Queued":
                    failures.append(ImageConversionFailure(item.input_path, "Cancelled before rendering."))
            reporter.skip_from(0)
        if not save_output:
            reporter.finish(cancelled=cancelled)
            return ImageBatchResult(successes, failures, cancelled, "", None)
        manifest, zip_path = _build_manifest_and_zip(
            stamp, options, successes, failures, cancelled, create_zip=create_zip and not cancelled,
            output_dir=destination, batch_diagnostics=reporter.diagnostics(final=True),
            controller=controller,
        )
        cancelled = controller.cancel.is_set()
        reporter.finish(cancelled=cancelled, manifest_path=manifest)
        return ImageBatchResult(successes, failures, cancelled, manifest, zip_path)
    except BaseException as exc:
        reporter.finish(cancelled=controller.cancel.is_set(), error=str(exc))
        raise
