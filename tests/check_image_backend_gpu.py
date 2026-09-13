import base64
import io
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.neural_rendering.image import api, batch, ui
from src.neural_rendering.image.models import ImageConversionOptions, NO_SAVE
from src.upscale.image import processor, ui as vsr_ui
from src.upscale.image.models import ImageUpscaleOptions, SETTING_FIELDS
from src.settings.models import UISettings, _validate
from src.settings.storage import save_settings, load_settings
from src.settings.presets import preset_document, import_settings_preset
import gradio as gr

source = Image.new('RGBA', (128, 96), (150, 100, 50, 180))
with io.BytesIO() as buffer:
    source.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
original_save = Image.Image.save
def memory_only_save(image, target, *args, **kwargs):
    assert hasattr(target, 'write'), f'Unexpected image file write: {target}'
    return original_save(image, target, *args, **kwargs)

for name, options, expected in [
    ('neural', dict(target_width=192, target_height=144, upscaling_factor=1.5, iterations=2), (192, 144)),
    ('vsr', dict(operation='vsr', target_width=192, target_height=144), (192, 144)),
]:
    print('RUN memory API', name, flush=True)
    with patch.object(Image.Image, 'save', memory_only_save), \
         patch.object(batch, 'OutputFile', side_effect=AssertionError('output file')), \
         patch.object(batch, 'prepare_output_dir', side_effect=AssertionError('output directory')), \
         patch.object(batch, '_build_manifest_and_zip', side_effect=AssertionError('manifest')), \
         patch.object(processor, 'OutputFile', side_effect=AssertionError('VSR output file')), \
         patch.object(processor, 'prepare_output_dir', side_effect=AssertionError('VSR output directory')):
        returned = api.enhance_image_memory(encoded, json.dumps(options), 'test-' + name)
    with Image.open(io.BytesIO(base64.b64decode(returned))) as output:
        assert output.size == expected
        assert output.getpixel((0, 0))[3] == 180
    assert not api._controllers
    print('PASS memory API with disk-image writes forbidden:', name, flush=True)

with tempfile.TemporaryDirectory() as temporary:
    folder = Path(temporary)
    input_path = folder / 'input.png'
    source.save(input_path)
    nr_args = [str(input_path), 'Default', 'Default', 1., 1., 1., -1., 1., 'Off', 'Default', NO_SAVE, 95, 'Auto', '_DLSS5', 1]
    print('RUN GUI no-save and PNG regression', flush=True)
    gallery, archive, rows, status = ui.render_image_batch(*nr_args, output_dir=folder / 'must-not-exist')
    assert len(gallery) == 1 and archive is None, status
    assert not (folder / 'must-not-exist').exists()
    assert rows[0][2] == 'Not saved' and 'No output files' in status
    vsr_options = ImageUpscaleOptions(scale_factor=1.5, output_format=NO_SAVE)
    values = [getattr(vsr_options, n) for n in SETTING_FIELDS]
    gallery, archive, rows, status = vsr_ui.render_image_batch([str(input_path)], *values, output_dir=folder / 'vsr-must-not-exist')
    assert len(gallery) == 1 and archive is None
    assert not (folder / 'vsr-must-not-exist').exists()
    nr_args[10] = 'PNG'
    gallery, archive, rows, status = ui.render_image_batch(*nr_args, output_dir=folder / 'saved')
    assert len(gallery) == 1 and archive and len(list((folder / 'saved').glob('*.png'))) == 1, status
    print('PASS GUI no-save previews in both tabs; PNG still saves an image and ZIP.', flush=True)
    settings = replace(UISettings(), image_format=NO_SAVE, upscale_image_output_format=NO_SAVE)
    _validate(settings)
    save_settings(folder / 'settings.ini', settings)
    loaded = load_settings(folder / 'settings.ini')
    assert loaded.image_format == loaded.upscale_image_output_format == NO_SAVE
    (folder / 'preset.json').write_text(json.dumps(preset_document('No save', settings)))
    imported = import_settings_preset(folder / 'preset.json', UISettings())[1]
    assert imported.image_format == imported.upscale_image_output_format == NO_SAVE
    with gr.Blocks():
        tab = ui.build_image_tab(UISettings())
        vsr_tab = vsr_ui.build_image_tab(UISettings())
    assert tab.output_format.value == vsr_tab.controls['output_format'].value == 'PNG'
    assert NO_SAVE in [v for label, v in tab.output_format.choices]
    print('PASS settings/preset round trip and both GUI PNG defaults.', flush=True)
print('ALL BACKEND CHECKS PASSED', flush=True)

