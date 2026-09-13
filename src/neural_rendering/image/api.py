"""Image-only API for VTS clients; uses the same renderer as the GUI."""
from __future__ import annotations

import base64
import io
import json
import threading

import gradio as gr
from PIL import Image, ImageOps

from ...core.jobs import Cancelled, JobController
from ...settings.storage import processing_gpu_settings
from .batch import _validate_options, convert_images
from .models import ImageConversionOptions, NO_SAVE
from ...upscale.image.models import ImageUpscaleOptions
from ...upscale.image.processor import upscale_image

_controllers: dict[str, JobController] = {}
_lock = threading.Lock()
_PARAMETERS = {
    "iterations", "nr_style", "nr_intensity", "local_tone_strength",
    "local_structure_strength", "skin_structure_strength", "automatic_mask",
    "target_width", "target_height", "operation", "vsr_quality", "nr_passes",
    "nr_color_strength", "tone_preservation", "face_skin_protection",
    "grain_preservation", "nr_gpu_mode",
}


def _render_image(image_source, parameters, request_id, progress):
    values = json.loads(parameters)
    if not isinstance(values, dict) or values.keys() - _PARAMETERS:
        raise gr.Error("Unknown image parameters. Use the VTS Merserk node's rendering settings.")
    operation = values.pop("operation", "neural")
    if operation not in {"neural", "vsr"}:
        raise gr.Error("Image operation must be neural or vsr.")
    vsr_quality = values.pop("vsr_quality", 4)
    width, height = values.pop("target_width", None), values.pop("target_height", None)
    if not request_id or len(request_id) > 128:
        raise gr.Error("A request ID is required.")
    if operation == "vsr" and values:
        raise gr.Error("Neural settings require the neural operation.")
    controller = JobController()
    with _lock:
        if request_id in _controllers:
            raise gr.Error("This request ID is already rendering.")
        _controllers[request_id] = controller
    current = None
    try:
        if isinstance(image_source, Image.Image):
            current = ImageOps.exif_transpose(image_source)
        else:
            with Image.open(image_source) as source:
                current = ImageOps.exif_transpose(source)
        if width is None and height is None:
            width, height = current.size
        if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 16384
               for n in (width, height)):
            raise gr.Error("Both target dimensions must be integers from 1 to 16384.")
        gpu_uuid = processing_gpu_settings()[0]
        neural_options = None
        if operation == "neural":
            neural_options = _validate_options(ImageConversionOptions(
                ai_gpu_uuid=gpu_uuid, output_format=NO_SAVE,
                target_width=width, target_height=height, **values))
        upscale_options = ImageUpscaleOptions(
            ai_gpu_uuid=gpu_uuid, vsr_quality=vsr_quality,
            size_mode="Custom dimensions", width=width, height=height,
            aspect_lock=False, output_format=NO_SAVE)
        upscale_options.validate()

        def check_cancel():
            if controller.cancel.is_set():
                raise Cancelled("Image rendering stopped by user.")

        def receive_image(image, name):
            nonlocal current
            replacement = image.copy()
            current.close()
            current = replacement

        # Shrink only the necessary axes; VSR handles every enlargement once.
        # VTS normally does this locally to avoid uploading unnecessary pixels.
        check_cancel()
        reduced_size = (min(current.width, width), min(current.height, height))
        if reduced_size != current.size:
            reduced = current.resize(reduced_size, Image.Resampling.LANCZOS)
            current.close()
            current = reduced
        needs_upscale = current.size != (width, height)
        if needs_upscale:
            upscale_image(current, upscale_options, controller=controller,
                          progress=lambda v, m: progress(v * (.35 if neural_options else 1), desc=m),
                          generate_previews=False, on_image=receive_image)
        check_cancel()
        if neural_options is not None:
            offset = .35 if needs_upscale else 0
            result = convert_images(
                [current], neural_options, controller=controller,
                progress=lambda v, m: progress(offset + v * (1 - offset), desc=m),
                generate_previews=False, create_zip=False, on_image=receive_image)
            if not result.successes:
                raise gr.Error(result.failures[0].error if result.failures else "Image rendering was cancelled.")
        check_cancel()
        final, current = current, None
        return final
    finally:
        if current is not None:
            current.close()
        with _lock:
            _controllers.pop(request_id, None)


def enhance_image(image_file: str, parameters: str, request_id: str, progress=gr.Progress()):
    # Retain the file API for existing clients; Gradio caches its uploads and returned preview.
    return _render_image(image_file, parameters, request_id, progress)


def enhance_image_memory(image_png: str, parameters: str, request_id: str, progress=gr.Progress()):
    """Base64 PNG transport avoids Gradio's uploaded/downloaded image cache."""
    with Image.open(io.BytesIO(base64.b64decode(image_png, validate=True))) as source:
        source.load()
        rendered = _render_image(source, parameters, request_id, progress)
    with rendered, io.BytesIO() as buffer:
        rendered.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def cancel_image(request_id: str):
    with _lock:
        controller = _controllers.get(request_id)
        if controller is not None:
            controller.stop()
    return controller is not None


def register_image_api():
    with gr.Column(visible=False):
        source = gr.File(type="filepath", file_types=["image"])
        parameters = gr.Textbox(value="{}")
        request_id = gr.Textbox()
        output = gr.Image(type="pil", format="png")
        run = gr.Button("VTS API render")
        cancel = gr.Button("VTS API cancel")
        cancelled = gr.Checkbox()
        memory_source = gr.Textbox()
        memory_output = gr.Textbox()
        memory_run = gr.Button("VTS API memory render")
    run.click(enhance_image, inputs=[source, parameters, request_id], outputs=output,
              api_name="vts_enhance", concurrency_limit=1)
    cancel.click(cancel_image, inputs=request_id, outputs=cancelled,
                 api_name="vts_cancel", queue=False)
    memory_run.click(enhance_image_memory, inputs=[memory_source, parameters, request_id],
                     outputs=memory_output, api_name="vts_enhance_memory", concurrency_limit=1)
