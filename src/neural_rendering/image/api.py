"""Image-only API for VTS clients; uses the same renderer as the GUI."""
from __future__ import annotations

import base64
import io
import json
import threading

import gradio as gr
from PIL import Image

from ...core.jobs import JobController
from ...settings.storage import processing_gpu_settings
from .batch import convert_images
from .models import ImageConversionOptions, NO_SAVE
from ...upscale.image.models import ImageUpscaleOptions
from ...upscale.image.processor import upscale_image

_controllers: dict[str, JobController] = {}
_lock = threading.Lock()
_PARAMETERS = {
    "upscaling_factor", "iterations", "nr_preset", "nr_style", "nr_intensity",
    "local_tone_strength", "local_structure_strength", "skin_structure_strength",
    "automatic_mask", "dlss_model_preset",
    "target_width", "target_height", "operation", "vsr_quality", "backend",
    "nr_passes", "nr_color_strength", "tone_preservation", "face_skin_protection",
    "grain_preservation", "mask_feather", "nr_gpu_mode",
}


def _render_image(image_source, parameters, request_id, progress):
    values = json.loads(parameters)
    if not isinstance(values, dict) or values.keys() - _PARAMETERS:
        raise gr.Error("Unknown image parameters. Use the VTS Merserk node's rendering settings.")
    operation = values.pop("operation", "neural")
    vsr_quality = values.pop("vsr_quality", 4)
    if operation not in {"neural", "vsr"}:
        raise gr.Error("Image operation must be neural or vsr.")
    backend = values.pop("backend", "auto")
    if backend not in {"auto", "neuroframe", "legacy"}:
        raise gr.Error("Backend must be auto, neuroframe or legacy.")
    legacy_needed = float(values.get("upscaling_factor", 1)) > 1 or any(
        values.get(name, "Default") != "Default" for name in ("nr_preset", "dlss_model_preset"))
    if operation == "neural" and values.get("target_width") is not None:
        if isinstance(image_source, Image.Image):
            source_size = image_source.size
        else:
            with Image.open(image_source) as source:
                source_size = source.size
        legacy_needed = legacy_needed or values["target_width"] > source_size[0] or values["target_height"] > source_size[1]
    if backend == "neuroframe" and legacy_needed:
        raise gr.Error("These scaling or preset settings require backend=legacy. Neuroframe supports native-size enhancement and downscaling; use the VSR operation for separate upscaling.")
    use_legacy = backend == "legacy" or (backend == "auto" and legacy_needed)
    renderer, option_class = convert_images, ImageConversionOptions
    if operation == "neural" and use_legacy:
        from ...legacy.images import convert_images as legacy_render
        from ...legacy.models import ImageConversionOptions as LegacyOptions
        renderer, option_class = legacy_render, LegacyOptions
        unsupported = set(values) - set(LegacyOptions.__dataclass_fields__)
        if unsupported:
            raise gr.Error("Legacy backend does not support: " + ", ".join(sorted(unsupported)))
    else:
        values.pop("nr_preset", None)
        values.pop("dlss_model_preset", None)
    gpu_uuid = processing_gpu_settings()[0]
    if operation == "vsr":
        width, height = values.get("target_width"), values.get("target_height")
        if width is None or height is None:
            raise gr.Error("RTX VSR requires both target_width and target_height.")
        options = ImageUpscaleOptions(ai_gpu_uuid=gpu_uuid, vsr_quality=vsr_quality,
                                      size_mode="Custom dimensions", width=width,
                                      height=height, aspect_lock=False, output_format=NO_SAVE)
    else:
        options = option_class(ai_gpu_uuid=gpu_uuid, output_format=NO_SAVE, **values)
    controller = JobController()
    if not request_id or len(request_id) > 128:
        raise gr.Error("A request ID is required.")
    with _lock:
        if request_id in _controllers:
            raise gr.Error("This request ID is already rendering.")
        _controllers[request_id] = controller
    try:
        rendered = []
        def receive_image(image, name):
            rendered.append(image.copy())
        report_progress = lambda value, message: progress(value, desc=message)
        if operation == "vsr":
            upscale_image(image_source, options, controller=controller, progress=report_progress,
                          generate_previews=False, on_image=receive_image)
        else:
            result = renderer(
                [image_source], options, controller=controller,
                progress=report_progress, generate_previews=False, create_zip=False,
                on_image=receive_image,
            )
            if not result.successes:
                raise gr.Error(result.failures[0].error if result.failures else "Image rendering was cancelled.")
        return rendered[0]
    finally:
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
