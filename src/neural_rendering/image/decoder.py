from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pillow_heif
import rawpy
import resvg_py
from PIL import Image, ImageCms, ImageOps

from ...core.paths import GRADIO_TEMP
from .models import RAW_EXTENSIONS

pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = 100_000_000

# Formats modern Chromium-based Gradio clients can display without a display
# conversion. Animated raster inputs are still converted below so the preview
# matches the processing pipeline's first-frame/page semantics.
_BROWSER_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp"}


@dataclass(slots=True)
class _DecodedImage:
    rgba: np.ndarray
    # None means the source is fully opaque. Avoid allocating/resampling a full
    # 8-bit alpha plane that contains only 255 values.
    alpha: np.ndarray | None
    decoder: str
    metadata: dict[str, object]
    warnings: list[str]


def _orientation_swaps_dimensions(image: Image.Image) -> bool:
    try:
        return int(image.getexif().get(274, 1)) in {5, 6, 7, 8}
    except (TypeError, ValueError):
        return False


def _open_pillow_source(path: Path) -> tuple[Image.Image, str]:
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            array = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
            )
        return Image.fromarray(array, mode="RGB"), "LibRaw"
    if suffix == ".svg":
        png = resvg_py.svg_to_bytes(svg_path=str(path), resources_dir=str(path.parent))
        return Image.open(io.BytesIO(png)), "resvg"
    return Image.open(path), "Pillow"


def probe_image(path: str | os.PathLike[str]) -> tuple[int, int]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    image, _decoder = _open_pillow_source(source)
    try:
        width, height = image.size
        if _orientation_swaps_dimensions(image):
            width, height = height, width
    finally:
        image.close()
    if width < 64 or height < 64:
        raise ValueError(
            f"{source.name} is {width}×{height}; DLSS requires both input dimensions to be at least 64 pixels."
        )
    return width, height


@lru_cache(maxsize=1)
def _srgb_profile() -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


@lru_cache(maxsize=1)
def _srgb_profile_bytes() -> bytes:
    return _srgb_profile().tobytes()


def initialize_image_runtime() -> None:
    """Initialize reusable image color-management state during app preparation."""
    _srgb_profile()
    _srgb_profile_bytes()


def _has_alpha(image: Image.Image, info: dict[str, object]) -> bool:
    return image.mode in {"LA", "RGBA"} or (image.mode == "P" and "transparency" in info)


def decode_image(path: str | os.PathLike[str]) -> _DecodedImage:
    source = Path("memory-image.png") if isinstance(path, Image.Image) else Path(path).resolve()
    image, decoder = (path.copy(), "Pillow memory") if isinstance(path, Image.Image) else _open_pillow_source(source)
    opened_image = image
    warnings: list[str] = []
    try:
        frame_count = int(getattr(image, "n_frames", 1) or 1)
        if frame_count > 1:
            warnings.append(f"Used the first frame/page of {frame_count}.")
            image.seek(0)
        original_mode = image.mode
        info = dict(image.info)
        source_format = image.format or source.suffix.lstrip(".").upper()
        image = ImageOps.exif_transpose(image)
        image.load()
        if original_mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
            warnings.append(f"Converted {original_mode} source data to 8-bit SDR sRGB.")

        has_alpha = _has_alpha(image, info)
        icc_profile = info.get("icc_profile")
        if isinstance(icc_profile, bytes) and icc_profile:
            # Color transforms operate on RGB only; preserve alpha independently.
            alpha_image = image.convert("RGBA").getchannel("A") if has_alpha else None
            rgb_image = image.convert("RGB")
            try:
                source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                rgb_image = ImageCms.profileToProfile(
                    rgb_image,
                    source_profile,
                    _srgb_profile(),
                    outputMode="RGB",
                )
            except (ImageCms.PyCMSError, OSError, ValueError) as exc:
                warnings.append(f"Embedded color profile could not be applied ({exc}); assumed sRGB.")
            rgb = np.asarray(rgb_image, dtype=np.uint8)
            rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = rgb
            if alpha_image is None:
                rgba[..., 3] = 255
                alpha = None
            else:
                alpha = np.ascontiguousarray(np.asarray(alpha_image, dtype=np.uint8))
                rgba[..., 3] = alpha
        else:
            # Common sRGB paths need only one Pillow conversion and one contiguous
            # NumPy allocation, instead of RGB + alpha + RGBA intermediates.
            rgba_image = image.convert("RGBA")
            rgba = np.array(rgba_image, dtype=np.uint8, copy=True)
            rgba_image.close()
            alpha = np.ascontiguousarray(rgba[..., 3]) if has_alpha else None

        exif = image.getexif()
        if exif:
            exif[274] = 1
        metadata: dict[str, object] = {
            "icc_profile": _srgb_profile_bytes(),
            "exif": exif.tobytes() if exif else None,
            "dpi": info.get("dpi"),
            "xmp": info.get("xmp"),
            "source_mode": original_mode,
            "source_format": source_format,
            "frame_count": frame_count,
        }
        return _DecodedImage(rgba, alpha, decoder, metadata, warnings)
    finally:
        image.close()
        if opened_image is not image:
            opened_image.close()


def decode_image_preview(
    path: str | os.PathLike[str], max_size: tuple[int, int] | None = (1200, 900)
) -> Image.Image:
    """Decode a UI preview without constructing the full render-time NumPy payload.

    ``max_size=None`` preserves the decoded source dimensions. This is used only
    by the opt-in full-size Gradio preview path.
    """
    source = Path(path).resolve()
    suffix = source.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        with rawpy.imread(str(source)) as raw:
            array = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
                half_size=max_size is not None,
            )
        image = Image.fromarray(array, mode="RGB")
        if max_size is not None:
            image.thumbnail(max_size, Image.Resampling.BILINEAR)
        return image.convert("RGBA")

    image, _decoder = _open_pillow_source(source)
    opened_image = image
    try:
        if int(getattr(image, "n_frames", 1) or 1) > 1:
            image.seek(0)
        if max_size is not None:
            # JPEG can ask its decoder for a reduced-resolution IDCT before loading.
            with __import__("contextlib").suppress(Exception):
                image.draft("RGB", max_size)
        info = dict(image.info)
        image = ImageOps.exif_transpose(image)
        if max_size is not None:
            image.thumbnail(max_size, Image.Resampling.BILINEAR)
        icc_profile = info.get("icc_profile")
        if isinstance(icc_profile, bytes) and icc_profile:
            has_alpha = _has_alpha(image, info)
            alpha = image.convert("RGBA").getchannel("A") if has_alpha else None
            rgb = image.convert("RGB")
            try:
                rgb = ImageCms.profileToProfile(
                    rgb,
                    ImageCms.ImageCmsProfile(io.BytesIO(icc_profile)),
                    _srgb_profile(),
                    outputMode="RGB",
                )
            except (ImageCms.PyCMSError, OSError, ValueError):
                pass
            result = rgb.convert("RGBA")
            if alpha is not None:
                result.putalpha(alpha)
            return result
        return image.convert("RGBA")
    finally:
        image.close()
        if opened_image is not image:
            opened_image.close()


def _direct_browser_image(source: Path) -> bool:
    suffix = source.suffix.lower()
    if suffix == ".svg":
        return True
    if suffix not in _BROWSER_IMAGE_EXTENSIONS:
        return False
    try:
        with Image.open(source) as image:
            return int(getattr(image, "n_frames", 1) or 1) <= 1
    except Exception:
        return False


def full_size_image_preview_path(path: str | os.PathLike[str]) -> str:
    """Return a browser-displayable full-resolution path for an image.

    Browser-native, single-frame formats are served directly so no pixels are
    copied or re-encoded. RAW/HEIF/TIFF/animated/other Pillow-readable sources
    are converted once to a lossless full-resolution PNG in the app-local
    Gradio temp directory.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if _direct_browser_image(source):
        return str(source)

    stat = source.stat()
    identity = f"{source}|{stat.st_dev}|{stat.st_ino}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8", "surrogatepass")).hexdigest()[:24]
    GRADIO_TEMP.mkdir(parents=True, exist_ok=True)
    destination = GRADIO_TEMP / f"dlss5-fullsize-source-{digest}.png"
    if destination.is_file():
        return str(destination.resolve())

    image = decode_image_preview(source, None)
    handle, raw_temp = tempfile.mkstemp(
        prefix="dlss5-fullsize-source-write-", suffix=".png", dir=GRADIO_TEMP
    )
    os.close(handle)
    temporary = Path(raw_temp)
    try:
        image.save(temporary, format="PNG", optimize=False, compress_level=1)
        os.replace(temporary, destination)
    finally:
        image.close()
        temporary.unlink(missing_ok=True)
    return str(destination.resolve())

