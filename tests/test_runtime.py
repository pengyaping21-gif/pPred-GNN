import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ppred.bundle import bundled_checkpoint
from ppred.runtime import fingerprint, inference_normalization, training_normalization


class RuntimeTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(task='single_rt', rt_min=None, rt_max=None, rt_unit='source_units', rt_normalization='auto')
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_continued_finetuning_preserves_embedded_rt_scale(self):
        saved = {'minimum': 16.6, 'maximum': 170.3, 'fit_split': 'train', 'unit': 'source_units'}
        scale = training_normalization(self.args(), [{'rt': 60.}, {'rt': 70.}],
                                       {'task': 'single_rt', 'normalization': saved})
        self.assertEqual((scale['minimum'], scale['maximum']), (16.6, 170.3))
        self.assertNotIn('origin', saved)

    def test_new_domain_fits_only_training_labels_when_requested(self):
        args = self.args(rt_normalization='train')
        scale = training_normalization(args, [{'rt': 60.}, {'rt': 70.}],
                                       {'task': 'single_rt', 'normalization': {'minimum': 0., 'maximum': 100.}})
        self.assertEqual((scale['minimum'], scale['maximum']), (60., 70.))
        self.assertEqual(scale['fit_split'], 'train')

    def test_checkpoint_policy_rejects_rt_task_transfer(self):
        with self.assertRaises(ValueError):
            training_normalization(self.args(task='crosslink_rt', rt_normalization='checkpoint'), [{'rt': 10.}],
                                   {'task': 'single_rt', 'normalization': {'minimum': 0., 'maximum': 100.}})

    def test_evaluation_rejects_wrong_task_and_nonfinite_scale(self):
        with self.assertRaises(ValueError):
            inference_normalization(self.args(), {'task': 'crosslink_rt'})
        with self.assertRaises(ValueError):
            inference_normalization(self.args(rt_min=float('nan'), rt_max=100.), {})

    def test_bundled_checkpoint_detects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'checkpoints').mkdir()
            weights = root / 'checkpoints/model.pth'
            weights.write_bytes(b'checkpoint bytes')
            entry = dict(fingerprint(weights), path='checkpoints/model.pth')
            manifest = root / 'checkpoints/manifest.json'
            manifest.write_text(json.dumps({'checkpoints': {'single_rt': entry}}))
            self.assertEqual(bundled_checkpoint('single_rt', root), weights)
            weights.write_bytes(b'changed bytes')
            with self.assertRaises(ValueError):
                bundled_checkpoint('single_rt', root)
            entry['path'] = '../outside.pth'
            manifest.write_text(json.dumps({'checkpoints': {'single_rt': entry}}))
            with self.assertRaises(ValueError):
                bundled_checkpoint('single_rt', root)


if __name__ == '__main__':
    unittest.main()
