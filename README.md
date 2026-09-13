> **Chargeuk fork:** Includes upstream Neuroframe improvements, VTS image and
> interpolation APIs, iterative enhancement and no-save controls. Existing DLSS
> scaling/preset workflows have an isolated compatibility backend.
> See [VTS changes, runtime setup and tests](docs/VTS_CHANGES.md).

# DLSS 5 Visual Enhancer

[![Downloads](https://img.shields.io/github/downloads/Merserk/dlss5-visual-enhancer/total.svg?style=flat-square&label=Downloads)](https://github.com/Merserk/dlss5-visual-enhancer/releases) ![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows11&logoColor=white) ![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX-76B900?style=flat-square&logo=nvidia&logoColor=white) ![DLSS](https://img.shields.io/badge/DLSS-5-76B900?style=flat-square) ![DLSS Frame Generation](https://img.shields.io/badge/DLSS-Frame%20Generation-76B900?style=flat-square&logo=nvidia&logoColor=white) ![RTX Video](https://img.shields.io/badge/RTX-Video-76B900?style=flat-square&logo=nvidia&logoColor=white) [![Patreon](https://img.shields.io/badge/Patreon-Merserk-FF424D?style=flat-square&logo=patreon&logoColor=white)](https://www.patreon.com/Merserk)

Portable Windows application for applying DLSS 5 Neural Rendering to images and videos through the **Neuroframe Engine**, NVIDIA DLSS Frame Generation to video frame interpolation, NVIDIA RTX Video Super Resolution and RTX Video HDR to dedicated Upscale workflows, and DLSS 5 Neural Rendering during Live video playback through a local Gradio interface. It is an independent community project and is not affiliated with, sponsored by, or endorsed by NVIDIA.

<img width="1810" height="1000" alt="Screenshot 2026-09-03 005153" src="https://github.com/user-attachments/assets/ad15df03-2934-4a20-90ca-5a6514a5de32" />

## Installation

1. Download the [latest release](https://github.com/Merserk/dlss5-visual-enhancer/releases/latest).
2. Unpack the downloaded ZIP archive.
3. Run `start.bat`.

### Original

https://github.com/user-attachments/assets/8df8bd4c-01b4-47dd-9705-3614a0b0ff75

### DLSS 5 Neural Rendering

https://github.com/user-attachments/assets/cff68783-4ee9-4c99-8b36-4eee2a6437ec

### Frame Generation (DLSSG)

https://github.com/user-attachments/assets/81c29005-e4f0-4acf-b9f7-d58850bb055f

## Main features

- **Neuroframe Engine:** new self-contained DLSS 5 Neural Rendering engine built around the NVIDIA NGX/DLSS runtime. Neural Rendering no longer depends on an external graphics injector or add-on. The engine supports a GPU-focused VRAM path and a RAM staging path that can be selected from Settings.
- **DLSS 5 Neural Rendering:** process images and videos with NR Style, NR Intensity, Local Tone Strength, Local Structure Strength, Skin Structure Strength, Automatic Mask, and the expanded v8.0 composition controls.
- **Multi Pass:** Neural Rendering can run from 1 to 4 passes, allowing additional DLSS 5 processing passes when stronger enhancement or refinement is wanted.
- **Neural composition controls:** NR Color Strength, Tone Preservation, Face/Skin Protection, Grain Preservation, and Mask Feather provide more control over how the DLSS result is blended with the original image.
- **Custom NR Mask:** upload a custom image mask to control where Neural Rendering is applied. The same selected mask is shared between Image, Video, and Live for the current application session.
- **Detail-Only:** one-click composition preset that sets NR Color Strength to 0 and Tone Preservation to 1, keeping the source color and tone while retaining Neural Rendering detail changes.
- **Shimmer Suppression:** Video and Live include temporal stabilization control for reducing unstable fine detail and shimmer between frames.
- **Neural Rendering scale:** Source, 75%, 50%, and 25% processing sizes are available. These settings control the resolution processed by Neural Rendering; resolution enlargement is handled separately by Upscale mode.
- **Real-time previews:** changing supported Neural Rendering controls on a single selected image or video can automatically refresh the preview. Images also provide a dedicated Preview action, while videos provide one-frame and three-second previews.
- **Image processing:** single-image and batch Neural Rendering with per-file progress, previews, diagnostic reports, direct single-file saving, and on-demand ZIP creation for multi-file batches.
- **Image formats:** common Pillow formats plus HEIF/HEIC, SVG, and many camera RAW formats. Neural Rendering outputs are PNG, JPEG, WebP, AVIF, or TIFF.
- **Image handling:** EXIF orientation is applied, embedded ICC input is converted to sRGB, supported EXIF/DPI/XMP metadata is retained, and alpha is preserved except when JPEG composites transparency over white. Animated and multipage sources use the first frame/page.
- **Video Neural Rendering:** single-video and batch-video processing with H.264, H.265, AV1, or ProRes Proxy output in MP4, MKV, or MOV where compatible. CPU and NVIDIA NVENC variants are available for H.264, H.265, and AV1; ProRes Proxy is CPU-based.
- **GPU video pipeline:** when the VRAM processing path and an NVIDIA NVENC output codec are used, supported Neural Rendering video jobs can keep more of the decode, DLSS, and encode path on the GPU to reduce unnecessary transfers through system memory.
- **HDR Mode:** Neural Rendering and Frame Interpolation can preserve HDR with a 10-bit output when H.265, AV1, or ProRes Proxy is selected. H.264 remains 8-bit SDR.
- **Upscale for images:** dedicated NVIDIA RTX Video Super Resolution workflow with VSR quality levels 1 to 4, 1× to 4× scaling or custom dimensions, aspect-ratio locking, image previewing, metadata preservation, and PNG/JPEG/WebP/AVIF/TIFF output.
- **Upscale for videos:** process SDR video with NVIDIA RTX Video Super Resolution, NVIDIA RTX Video HDR, or both in the same workflow. VSR supports 1× to 4× scaling or custom dimensions, while RTX Video HDR provides adjustable contrast, saturation, middle gray, peak luminance up to 2000 nits, and selectable HDR processing precision.
- **Frame Interpolation:** single-video and batch-video processing with NVIDIA DLSS Frame Generation, selectable output rates from 23.976 to 480 FPS, Auto/Native DLSSG/Cascade processing, and a three-second preview for a single selected video.
- **Live:** watch local videos, direct network streams, YouTube, and Twitch with DLSS 5 Neural Rendering applied during playback. Neural Rendering effect changes can be applied while the Live session is running without restarting playback processing.
- **Live real-time controls:** NR Style, NR Intensity, NR Passes, tone/structure controls, composition controls, Shimmer Suppression, Automatic Mask, Custom NR Mask, and Processing Engine Path changes can be applied to active processing. Scale and source/playback configuration take effect on the next Start.
- **Live playback controls:** Source quality, processing height from 480p to 2160p, Auto/Source/60/30/24 FPS modes, 1/2/4-second segments, adjustable playback buffering, and optional automatic playback in the bundled MPV player.
- **Direct disk processing:** Neural Rendering Image/Video, Upscale Image/Video, and Frame Interpolation support an absolute Input path for a file or folder and an optional Output path. Folder processing uses supported files from that folder only, sorted by filename. Direct disk mode disables browser previews/downloads and writes results directly to disk.
- **Batch progress:** every file has its own Queued, Running, Complete, Failed, Cancelled, or Skipped state with progress percentage, elapsed time, output path, and processing details. Completed outputs are preserved if a batch is stopped while incomplete work is cleaned up safely.
- **Simplified saving:** a completed single-file job exposes a direct Save Image or Save Video action. Multi-file jobs expose Save as ZIP, and the ZIP is created only when requested instead of being built automatically after rendering.
- **Video previews:** input/output players are used for single uploaded videos and hidden for multi-video batches or direct disk mode. Preview Encoding controls whether the actual result or a browser-compatible H.264 preview is shown.
- **Media preservation:** frame timestamps and display rotation are handled; original metadata and chapters are copied where supported. MKV can copy compatible audio/subtitle streams, while MP4/MOV use AAC audio. Supported text subtitles can be retained in workflows that enable subtitle preservation.
- **GPU selection:** AI Processing and Video Processing GPUs can be selected separately. The AI GPU is used by DLSS Neural Rendering, DLSS Frame Generation, and RTX Video processing; the Video GPU is used for codecs suffixed `(NVIDIA NVENC)`. Automatic selection is available for both.
- **GPU compatibility:** the application detects NVIDIA RTX hardware first and lets the installed NVIDIA runtime and driver determine actual DLSS, Frame Generation, and RTX Video feature availability instead of relying on a fixed architecture allowlist.
- **Preview Encoding:** Auto uses the result directly when it is browser-playable and otherwise creates a compatible H.264 preview; Always H.264 always creates a browser-compatible preview; Disabled sends the actual output to the browser without creating a compatibility preview.
- **Full size image previews:** Settings can switch image previews from the normal bounded preview size to full-resolution quality previews for closer inspection.
- **Render metadata:** supported Neural Rendering image and video outputs can store applied DLSS 5 settings, including v8.0 Multi Pass, composition, mask, scale, and processing-path information, without intentionally replacing an existing user description.
- **Renaming:** Neural Rendering, Upscale, and Frame Interpolation support Auto, Copy, and Custom naming. Custom defaults use `_Neural_Rendering`, `_Upscale`, and `_Frame_Interpolation`, and existing output files are never silently overwritten.
- **Safety and diagnostics:** GPU jobs are serialized, Stop cancels active workers and encoders, completed batch outputs are retained, incomplete outputs are cleaned up, and final files are published only after the relevant output checks succeed. Session logs and per-job diagnostic reports are kept for troubleshooting.
- **Persistent controls:** shared Neural Rendering controls, Frame Interpolation settings, Upscale settings, GPU selection, Processing Engine Path, HDR selections, Preview Encoding, Full size quality preview, and naming/output settings are stored in `config.ini`. Settings presets can be exported to or imported from JSON. Live source, quality, buffering, and playback controls remain session-local.

The application creates `outputs/`, `logs/`, `jobs/`, `live/`, `config/`, and `temp/gradio/` as needed. Successful media is written to `outputs/`, logs and diagnostic information are written under `logs/`, active render work uses `jobs/`, Live uses temporary session data under `live/`, persistent application settings are stored under `config/`, and Gradio uploads/previews use the application-local `temp/gradio/` cache. Stale application cache and abandoned job data are cleaned up automatically without touching completed outputs or logs.

## Requirements

- 64-bit Windows 11 with Direct3D 12.
- NVIDIA GeForce RTX GPU with a compatible NVIDIA driver. The application identifies RTX hardware and lets the bundled NVIDIA runtimes determine actual feature support.
- **DLSS 5 Neural Rendering:** supported on NVIDIA GeForce RTX 20, RTX 30, RTX 40, and RTX 50 Series GPUs, including other desktop, laptop, and workstation variants based on the same Turing, Ampere, Ada Lovelace, and Blackwell architectures. A compatible NVIDIA driver and the bundled DLSS Neural Rendering runtime are required.
- **Frame Interpolation:** supported on NVIDIA GeForce RTX 40 and RTX 50 Series GPUs, including other desktop, laptop, and workstation variants based on the same Ada Lovelace and Blackwell architectures. A compatible NVIDIA driver and DLSS Frame Generation runtime are required. Hardware-accelerated GPU scheduling (HAGS) should be enabled; if it is disabled, the app reports it diagnostically and lets the runtime determine whether Frame Generation can start.
- RTX Video Upscale requires a GPU and driver combination that reports the requested NVIDIA RTX Video Super Resolution and/or RTX Video HDR capability. Video Upscale currently accepts SDR source video; RTX Video HDR creates a new HDR output from that SDR source.
- NVIDIA NVENC output requires working hardware encoder support on the selected Video Processing GPU for the chosen codec and output size. Plain H.264/H.265/AV1 and ProRes Proxy encoding remain CPU-based.
- Live local files and direct stream URLs use the portable media tools included with the application. YouTube and Twitch page URLs require the bundled `yt-dlp`, while automatic playback uses the bundled MPV player and can be disabled with **Open in MPV**.

## Settings

### Neural Rendering

| Neural control | Values | Default |
| --- | --- | --- |
| NR Style | Default, Natural, Cinematic | Default |
| Scale | Source (Original), 75%, 50%, 25% | Source (Original) |
| NR Intensity | 0.00–2.00 | 1.00 |
| NR Passes | 1–4 | 1 |
| Local Tone Strength | 0.00–2.00 | 1.00 |
| Local Structure Strength | 0.00–2.00 | 1.00 |
| Skin Structure Strength | -1.00–2.00 | -1.00 |
| NR Color Strength | 0.00–1.00 | 1.00 |
| Tone Preservation | 0.00–1.00 | 0.00 |
| Face/Skin Protection | 0.00–1.00 | 0.00 |
| Grain Preservation | 0.00–1.00 | 0.00 |
| Mask Feather | 0–128 output pixels | 0 |
| Automatic Mask | Off, On | Off |
| Shimmer Suppression | 0.00–1.00; Video and Live only | 0.70 |

**NR Passes** controls how many Neural Rendering passes are applied, from one to four. Additional passes increase the amount of processing and can produce a stronger cumulative result.

**Scale** controls the resolution entering Neural Rendering. Source keeps the original dimensions; 75%, 50%, and 25% perform a Lanczos downscale before DLSS processing. Neural Rendering output is limited to the supported 7680×4320 boundary and requires both processed dimensions to remain at least 64 pixels.

**Custom NR Mask** accepts an uploaded image mask and is shared between the Neural Rendering Image, Video, and Live interfaces for the current session. **Mask Feather** softens the transition around the selected mask in output pixels.

**Detail-Only** is a convenience preset that sets NR Color Strength to `0.00` and Tone Preservation to `1.00`. The remaining Neural Rendering controls stay editable.

Changing Skin Structure Strength above its native `-1.00` value automatically enables Automatic Mask so that the skin adjustment can take effect. Automatic Mask can still be switched Off manually afterward.

### Upscale: Image

| Image Upscale setting | Choices and behavior | Default |
| --- | --- | --- |
| VSR quality | 1 Low, 2 Medium, 3 High, 4 Ultra | 4 Ultra |
| Output sizing | Scale factor or Custom dimensions | Scale factor |
| Scale factor | 1×, 1.5×, 2×, 3×, or 4× | 2× |
| Custom dimensions | Up to 16384 pixels per dimension; output cannot be smaller than the source | 3840×2160 |
| Lock aspect ratio | Off, On | On |
| Output format | PNG, JPEG, WebP, AVIF, TIFF | PNG |
| Image quality | 1–100 for lossy formats | 95 |
| Preserve metadata | Off, On | On |
| Rename | Auto, Copy, Custom | Auto |

RTX Video Super Resolution can also be used at **1×** for native-resolution enhancement without enlarging the image. A single selected image can be previewed before the final upscale is saved.

### Upscale: Video

| Video Upscale setting | Choices and behavior | Default |
| --- | --- | --- |
| RTX Video Super Resolution | Off, On | On |
| VSR quality | 1 Low, 2 Medium, 3 High, 4 Ultra | 4 Ultra |
| Output sizing | Scale factor or Custom dimensions | Scale factor |
| Scale factor | 1×, 1.5×, 2×, 3×, or 4× | 2× |
| Custom dimensions | Up to 16384 pixels per dimension; output cannot be smaller than the source when VSR is enabled | 3840×2160 |
| Lock aspect ratio | Off, On | On |
| RTX Video HDR | Off, On; converts SDR input to HDR | Off |
| HDR contrast | 0–200 | 100 |
| HDR saturation | 0–200 | 100 |
| HDR middle gray | 10–100 | 50 |
| HDR peak luminance | 400–2000 nits | 1000 nits |
| HDR processing precision | Packed 10-bit, Packed 10-bit (FP16) | Packed 10-bit |
| Video codec | H.264, H.265, AV1, ProRes Proxy; CPU/NVENC variants where available | H.265 (NVIDIA NVENC) |
| Container | MP4, MKV, MOV; ProRes Proxy requires MKV or MOV | MP4 |
| Encoding quality | Auto (Default), Good, Best, Max | Auto (Default) |
| Rename | Auto, Copy, Custom | Auto |

RTX Video Super Resolution and RTX Video HDR can be used together in one processing workflow. VSR can also be disabled when only SDR-to-HDR conversion is wanted. At least one of VSR or RTX Video HDR must be enabled.

Video Upscale accepts SDR source footage. Existing HDR input is rejected rather than being silently reprocessed as SDR. RTX Video HDR output requires H.265, AV1, or ProRes Proxy because H.264 cannot store the generated HDR output used by this workflow.

### Frame Interpolation

| Frame Interpolation setting | Choices and behavior | Default |
| --- | --- | --- |
| Output FPS | 23.976, 25, 29.97, 30, 50, 59.94, 60, 90, 119.88, 120, 144, 165, 180, 240, 360, or 480 FPS | 60 |
| DLSS engine | Auto, Native DLSSG, Cascade | Auto |
| Video codec | H.264, H.265, AV1, or ProRes Proxy, with CPU/NVENC variants where available | H.264 |
| Container | MP4, MKV, MOV; ProRes Proxy requires MKV or MOV | MP4 |
| Encoding quality | Auto (Default), Good, Best, Max | Auto (Default) |
| HDR Mode | Off, On; 10-bit output that copies the input colorspace; H.265/AV1/ProRes only | Off |
| Rename | Auto, Copy, Custom | Auto |

`Auto` uses an exact native DLSSG grid when the requested source-to-output rate is directly supported and uses the Cascade path when needed. If the selected output FPS is equal to or below the source rate, source frames are resampled without generating additional frames.

### Live

| Live setting | Choices and behavior | Default |
| --- | --- | --- |
| Source | Online URL or Local video | Online |
| Source quality | Auto, 480p, 720p, 1080p, 1440p, 2160p | Auto |
| Max input height | 480p, 720p, 1080p, 1440p, 2160p before Neural Rendering | 720p |
| Segment length | 1, 2, or 4 seconds | 2 seconds |
| Live frame rate | Auto, Source, 60, 30, or 24 FPS | Auto |
| Playback buffer | 2–30 seconds | 6 seconds |
| Open in MPV | Off, On | On |

Live shares the Neural Rendering controls with the Image and Video workflows. Effect settings including Multi Pass, composition controls, Shimmer Suppression, Automatic Mask, Custom NR Mask, and Processing Engine Path can update during an active session. Buffered video retains the previous appearance until playback reaches frames processed with the newly applied settings.

Scale, source selection, source quality, maximum input height, segment length, target frame rate, playback buffer, and MPV launch behavior are fixed for the current run and take effect on the next **Start Live**.

### Neural Rendering and Frame Interpolation output

| Output setting | Choices and behavior |
| --- | --- |
| Image format | PNG/TIFF are lossless; JPEG/WebP/AVIF use the 1–100 quality control |
| Video codec | H.264, H.265, AV1, or ProRes Proxy, with CPU/NVENC variants where available |
| Container | MP4, MKV, MOV; ProRes Proxy requires MKV or MOV |
| Encoding quality | Auto (Default) uses resolution/FPS/codec; Good and Best increase the automatic target; Max uses constant-quality/lossless-style codec settings where supported |
| HDR Mode | Off, On; 10-bit output that preserves source colorspace for H.265/AV1/ProRes |
| Rename | Auto creates a workflow-specific timestamped name; Copy keeps the source base name; Custom appends the entered suffix |

Plain H.264/H.265/AV1 use CPU encoders. Only codecs suffixed `(NVIDIA NVENC)` require working NVENC support on the selected Video Processing GPU. ProRes Proxy uses CPU-based 10-bit 4:2:2 encoding.

### Application

| Application setting | Choices and behavior | Default |
| --- | --- | --- |
| AI Processing GPU | Automatic or a compatible NVIDIA RTX GPU used for Neural Rendering, Frame Generation, and RTX Video processing | Automatic |
| Video Processing GPU | Automatic or an available NVIDIA GPU used for codecs suffixed `(NVIDIA NVENC)` | Automatic |
| Processing Engine Path | VRAM keeps the supported Neural Rendering path GPU-focused; RAM stages frames through system memory | VRAM |
| Preview Encoding | Auto, Always H.264, Disabled | Auto |
| Full size quality preview | Off uses bounded image previews; On keeps full-resolution image previews where supported | Off |
| Settings preset | Export or import adjustable application settings as JSON | n/a |

Saved GPU selections use stable GPU identity. If a previously saved GPU is no longer available, that selection returns to Automatic instead of silently selecting a different saved device.

Settings presets include the current v8.0 Neural Rendering, Multi Pass, composition, processing-path, Upscale, Frame Interpolation, GPU, HDR, preview, output, and naming controls. Older compatible preset files are migrated to the current settings schema when imported.

## License and third-party notices

Original application code is licensed under the MIT License, copyright © 2026 Merserk. That license covers only original project code; it does not relicense or grant rights to any third-party software, model, binary, trademark, media, or other asset.

- **NVIDIA DLSS/NGX and RTX Video:** NVIDIA and its suppliers retain their rights in genuine NVIDIA SDK and runtime files used for DLSS Neural Rendering, DLSS Frame Generation, and RTX Video features. Use and distribution are governed by the applicable NVIDIA license terms, including the [NVIDIA RTX SDK License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt) where applicable. Their presence in a portable package does not imply a standalone redistribution right, and this project must not be represented as NVIDIA-sponsored or endorsed.
- **FFmpeg:** FFmpeg and the bundled FFmpeg build retain their own copyright and license terms. Anyone redistributing the included binaries must preserve the applicable notices and satisfy the license and corresponding-source obligations of that build. See [FFmpeg licensing](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md).
- **MPV and yt-dlp:** Live mode can use the bundled portable MPV player and yt-dlp resolver. They and their dependencies retain their own copyright and license terms; preserve the notices shipped with each distribution.
- **Python and packages:** Python is provided under the [PSF License](https://docs.python.org/3.13/license.html). The portable Python runtime and packages including Gradio, Pillow, pillow-heif, rawpy, resvg-py, PyAV, OpenCV, NumPy, their transitive dependencies, and bundled codecs retain their own copyright and license terms; preserve the notices shipped with each distribution.

NVIDIA, GeForce RTX, NGX, DLSS, and RTX Video are trademarks and/or registered trademarks of NVIDIA Corporation. FFmpeg, MPV, yt-dlp, Python, and other names belong to their respective owners. Codec patent or other permissions may also be required depending on jurisdiction and use. Review the controlling licenses before building or distributing a complete package.
