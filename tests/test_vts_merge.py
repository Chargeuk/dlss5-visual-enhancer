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
from src.legacy import images as legacy
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

    def test_auto_uses_modern_renderer_for_compatible_vts_settings(self):
        with patch.object(api, 'convert_images', self.renderer):
            result = api._render_image(self.source, json.dumps(dict(nr_preset='Default', dlss_model_preset='Default', iterations=3)), 'modern-test', lambda *a, **k: None)
        self.assertEqual(self.options.iterations, 3)
        self.assertEqual(self.options.output_format, NO_SAVE)
        self.assertEqual(result.size, self.source.size)
        self.assertEqual(api._controllers, {})

    def test_auto_preserves_legacy_scaling_and_presets(self):
        for values in (dict(upscaling_factor=1.5), dict(nr_preset='Preset #2'), dict(dlss_model_preset='K')):
            with self.subTest(values=values), patch.object(legacy, 'convert_images', self.renderer):
                result = api._render_image(self.source, json.dumps(values), 'legacy-test', lambda *a, **k: None)
            self.assertEqual(result.size, self.source.size)
            for key, value in values.items(): self.assertEqual(getattr(self.options, key), value)

    def test_explicit_backend_never_silently_drops_unsupported_controls(self):
        for values in (dict(backend='neuroframe', dlss_model_preset='K'), dict(backend='legacy', nr_passes=2)):
            with self.subTest(values=values), self.assertRaises(gr.Error):
                api._render_image(self.source, json.dumps(values), 'invalid-test', lambda *a, **k: None)

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
