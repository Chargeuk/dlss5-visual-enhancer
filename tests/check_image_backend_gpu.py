import base64
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.neural_rendering.image import api, batch, ui
from src.upscale.image import processor, ui as vsr_ui
from src.settings.models import UISettings
from src.settings.storage import save_settings, load_settings
import gradio as gr

source = Image.new('RGBA', (128, 96), (150, 100, 50, 180))
with io.BytesIO() as buffer:
    source.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
original_save = Image.Image.save

def memory_only(image, target, *args, **kwargs):
    assert hasattr(target, 'write'), f'Unexpected image file write: {target}'
    return original_save(image, target, *args, **kwargs)

for name, options, expected in [
    ('neuroframe', dict(iterations=2, nr_passes=2), (128, 96)),
    ('vsr-then-neuroframe', dict(target_width=192, target_height=144, iterations=3, nr_passes=2, nr_color_strength=.8, tone_preservation=.3, face_skin_protection=.2, grain_preservation=.1), (192, 144)),
    ('mixed-resize', dict(target_width=96, target_height=144, iterations=2), (96, 144)),
    ('vsr', dict(operation='vsr', target_width=192, target_height=144), (192, 144)),
]:
    print('RUN memory API', name, flush=True)
    with patch.object(Image.Image, 'save', memory_only), \
         patch.object(batch, 'OutputFile', side_effect=AssertionError('NR file write')), \
         patch.object(batch, 'prepare_output_dir', side_effect=AssertionError('NR directory')), \
         patch.object(processor, 'OutputFile', side_effect=AssertionError('VSR file write')):
        returned = api.enhance_image_memory(encoded, json.dumps(options), 'check-' + name)
    with Image.open(io.BytesIO(base64.b64decode(returned))) as output:
        assert output.size == expected, output.size
        assert output.getpixel((0, 0))[3] == 180
    assert not api._controllers
    print('PASS', name, flush=True)

with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory)
    image_path = folder / 'input.png'
    source.save(image_path)
    # Match the new upstream GUI's full input contract, plus our iterations.
    values = [str(image_path), 'Default', 1., 1, 1., 1., -1., 1., 'Off',
              1., 0., 0., 0., 0, None, True, 'Do not save', 95, 'Auto', '_NR', 2]
    gallery, files, rows, status = ui.render_image_batch(*values, output_dir=folder / 'no-save')
    assert len(gallery) == 1 and not files and rows[0][2] == 'Not saved', status
    assert not (folder / 'no-save').exists()
    values[16] = 'PNG'
    gallery, files, rows, status = ui.render_image_batch(*values, output_dir=folder / 'saved')
    assert len(gallery) == len(files) == 1, status
    assert len(list((folder / 'saved').glob('*.png'))) == 1
    print('PASS GUI no-save and final PNG output', flush=True)
print('ALL IMAGE BACKEND GPU CHECKS PASSED', flush=True)
