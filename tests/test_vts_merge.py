import base64
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.neural_rendering.image import api, batch, ui
from src.neural_rendering.image.models import ImageConversionOptions, NO_SAVE
from src.settings.models import UISettings, _validate
from src.settings.storage import load_settings, save_settings
from src.settings.presets import preset_document, import_settings_preset
from src.core.jobs import JobController, Cancelled
import gradio as gr


class Session:
    instances = []
    def __init__(self, **kwargs):
        self.render_width = kwargs['output_width']
        self.render_height = kwargs['output_height']
        self.closed = False
        self.bridge_logs = []
        self.diagnostics = SimpleNamespace(memory_path='test')
        self.inputs = []
        self.instances.append(self)
    def process(self, *, index, rgba, reset, pts):
        self.inputs.append(rgba.copy())
        result = rgba.copy()
        result[..., 0] += 1
        return result, pts
    def structured_status(self): return {}
    def close(self): self.closed = True
    def abort(self): self.closed = True


class MergeTests(unittest.TestCase):
    def setUp(self):
        Session.instances = []
        self.source = Image.new('RGBA', (128, 96), (10, 20, 30, 180))

    def renderer(self, sources, options, **kwargs):
        self.options = options
        kwargs['on_image'](sources[0], 'test.png')
        return SimpleNamespace(successes=[object()])

    def test_modern_renderer_receives_both_loops_and_controls(self):
        values = dict(iterations=3, nr_passes=2, nr_color_strength=.6,
                      tone_preservation=.3, face_skin_protection=.4, grain_preservation=.2)
        with patch.object(api, 'convert_images', self.renderer), \
             patch.object(api, 'upscale_image', side_effect=AssertionError('unneeded upscale')):
            result = api._render_image(self.source, json.dumps(values), 'modern-test', lambda *a, **k: None)
        for key, value in values.items(): self.assertEqual(getattr(self.options, key), value)
        self.assertEqual(self.options.output_format, NO_SAVE)
        self.assertEqual(result.size, self.source.size)
        self.assertEqual(api._controllers, {})

    def test_upscale_once_then_neural_at_exact_size(self):
        events = []
        def upscale(source, options, **kwargs):
            events.append(('vsr', source.size, options.vsr_quality))
            with source.resize((options.width, options.height)) as enlarged:
                enlarged.putpixel((0, 0), (90, 20, 30, 180))
                kwargs['on_image'](enlarged, 'upscaled')
        def render(sources, options, **kwargs):
            events.append(('nr', sources[0].size, sources[0].getpixel((0,0))[0]))
            return self.renderer(sources, options, **kwargs)
        with patch.object(api, 'upscale_image', upscale), patch.object(api, 'convert_images', render):
            result = api._render_image(self.source, json.dumps(dict(target_width=192, target_height=144,
                       vsr_quality=3, iterations=3, nr_passes=2)), 'combined-test', lambda *a, **k: None)
        self.assertEqual(events, [('vsr', (128, 96), 3), ('nr', (192,144), 90)])
        self.assertEqual((self.options.iterations, self.options.nr_passes), (3,2))
        self.assertEqual(self.options.upscaling_factor, 1)
        self.assertEqual(result.size, (192,144))

    def test_mixed_resize_shrinks_only_one_axis_before_vsr(self):
        seen = []
        def upscale(source, options, **kwargs):
            seen.append(source.size)
            with source.resize((options.width,options.height)) as final:
                kwargs['on_image'](final, 'test')
        with patch.object(api, 'upscale_image', upscale), \
             patch.object(api, 'convert_images', side_effect=AssertionError('neural disabled')):
            result = api._render_image(self.source, json.dumps(dict(operation='vsr', target_width=64,
                        target_height=192)), 'mixed-test', lambda *a, **k: None)
        self.assertEqual(seen, [(64,96)])
        self.assertEqual(result.size, (64,192))

    def test_invalid_dimensions_or_passes_fail_before_upscaling(self):
        for values in (dict(target_width=192), dict(target_width=192, target_height=144, nr_passes=5)):
            with self.subTest(values=values), patch.object(api, 'upscale_image', side_effect=AssertionError('GPU work')):
                with self.assertRaises((ValueError, gr.Error)):
                    api._render_image(self.source, json.dumps(values), 'invalid-test', lambda *a, **k: None)
            self.assertEqual(api._controllers, {})

    def test_cancellation_between_stages_skips_neural(self):
        def upscale(source, options, **kwargs):
            kwargs['on_image'](source, 'test')
            kwargs['controller'].stop()
        with patch.object(api, 'upscale_image', upscale), \
             patch.object(api, 'convert_images', side_effect=AssertionError('neural after cancellation')):
            with self.assertRaises(Cancelled):
                api._render_image(self.source, json.dumps(dict(target_width=192,target_height=144)),
                                  'cancel-stages', lambda *a, **k: None)
        self.assertEqual(api._controllers, {})

    def test_cancel_only_targets_its_request(self):
        first, second = JobController(), JobController()
        with patch.dict(api._controllers, {'first': first, 'second': second}, clear=True):
            self.assertTrue(api.cancel_image('first'))
            self.assertTrue(first.cancel.is_set())
            self.assertFalse(second.cancel.is_set())
            self.assertFalse(api.cancel_image('absent'))

    def test_settings_and_preset_preserve_iterations_and_no_save(self):
        settings = replace(UISettings(), image_iterations=7, nr_passes=3, image_format=NO_SAVE, upscale_image_output_format=NO_SAVE)
        _validate(settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.ini'
            save_settings(path, settings)
            loaded = load_settings(path)
            self.assertEqual((loaded.image_iterations, loaded.nr_passes, loaded.image_format, loaded.upscale_image_output_format), (7, 3, NO_SAVE, NO_SAVE))
            preset = Path(directory) / 'preset.json'
            preset.write_text(json.dumps(preset_document('Merge', settings)))
            imported = import_settings_preset(preset, UISettings())[1]
            self.assertEqual(imported.image_iterations, 7)

    def test_iterations_feed_previous_output_and_only_deliver_final_image(self):
        delivered = []
        options = ImageConversionOptions(iterations=3, target_width=96, target_height=64, output_format=NO_SAVE)
        with patch.object(batch, 'prepare_runtime', return_value=SimpleNamespace(gpus=(), runtime_bundle={})), \
             patch.object(batch, 'resolve_runtime_ai_gpu', return_value={'display_name':'test'}), \
             patch.object(batch, 'DLSSFrameSession', Session), \
             patch.object(batch, 'snapshot_session', return_value=SimpleNamespace()), \
             patch.object(batch, 'verify_feature_18', return_value={}), \
             patch.object(batch, 'prepare_output_dir', side_effect=AssertionError('directory write')), \
             patch.object(batch, 'OutputFile', side_effect=AssertionError('file write')), \
             patch.object(batch, '_build_manifest', side_effect=AssertionError('manifest write')):
            result = batch.convert_images([self.source, self.source], options, on_image=lambda image, name: delivered.append(image.copy()))
        self.assertEqual(len(result.successes), 2, result.failures)
        self.assertEqual(len(delivered), 2)
        self.assertTrue(all(image.size == (96, 64) and image.getpixel((48,32))[0] == 13 for image in delivered))
        self.assertTrue(all(session.closed for session in Session.instances))
        self.assertEqual(len(Session.instances), 1)
        self.assertEqual(len(Session.instances[0].inputs), 6)

    def test_no_save_respects_cancellation_from_output_callback(self):
        controller = JobController()
        options = ImageConversionOptions(output_format=NO_SAVE)
        with patch.object(batch, 'prepare_runtime', return_value=SimpleNamespace(gpus=(), runtime_bundle={})), \
             patch.object(batch, 'resolve_runtime_ai_gpu', return_value={'display_name':'test'}), \
             patch.object(batch, 'DLSSFrameSession', Session), \
             patch.object(batch, 'snapshot_session', return_value=SimpleNamespace()), \
             patch.object(batch, 'verify_feature_18', return_value={}):
            result = batch.convert_images([self.source], options, controller=controller,
                                          on_image=lambda *args: controller.stop())
        self.assertTrue(result.cancelled)
        self.assertFalse(result.successes)


if __name__ == '__main__': unittest.main()
