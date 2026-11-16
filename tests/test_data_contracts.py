import tempfile
from pathlib import Path
import unittest
import json
import subprocess
import sys

import torch

from ppred.chemistry import composition, crosslink_structure, single_structure
from ppred.data import LengthBatchSampler, check_splits, read_records
from ppred.runtime import make_loader


class DataContractTests(unittest.TestCase):
    def test_splitter_reports_exclusions_and_keeps_groups_disjoint(self):
        def block(sequence, title):
            return ('BEGIN IONS\nTITLE=' + title + '\nSEQ=' + sequence + '@\n'
                    'CHARGE=2+\nNCE=27\nINSTRUMENT_NAME=QE\n'
                    'ACTIVATION_TYPE=HCD\nINSTRUMENT_TYPE=FT\n'
                    '100 100 b1-z1\nEND IONS\n')
        valid = [block(sequence, str(index)) for index, sequence in enumerate(['ACDK', 'PEPTIDEK', 'AAAAK'])]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mgf'
            source.write_text(''.join(valid + [valid[0], block('A' * 61, 'long')]))
            output = Path(directory) / 'split'
            script = Path(__file__).resolve().parents[1] / 'scripts/split_data.py'
            subprocess.run([sys.executable, str(script), '--task', 'ms2', '--input', str(source),
                            '--output', str(output), '--skip-invalid'], check=True, capture_output=True)
            result = json.loads((output / 'split_summary.json').read_text())
            self.assertEqual(result['excluded_records'], 1)
            self.assertEqual(result['exact_duplicates_removed'], 1)
            partitions = {name: read_records(output / (name + '.mgf'), 'ms2') for name in ['train', 'val', 'test']}
            self.assertEqual(sum(len(rows) for rows in partitions.values()), 3)
            check_splits(partitions, 'ms2')

    def test_batch_record_ids_are_not_incremented_as_graph_indices(self):
        rows = [{'seq': 'ACDK', 'mod': '', 'rt': 10., 'linker': ''},
                {'seq': 'PEPTIDEK', 'mod': '', 'rt': 20., 'linker': ''}]
        loader = make_loader(rows, 'single_rt', {'minimum': 0., 'maximum': 30.}, 2)
        batch = next(iter(loader))
        self.assertEqual(batch.record_id.tolist(), [0, 1])

    def test_carbamidomethyl_element_difference(self):
        _, unmodified = single_structure('ACDK', '')
        _, modified = single_structure('ACDK', '2,Carbamidomethyl[C];')
        difference = composition(modified)[1] - composition(unmodified)[1]
        # Element order starts C,H,O,N,P,S; carbamidomethyl adds C2H3NO.
        self.assertEqual(difference[:6].tolist(), [2., 3., 1., 1., 0., 0.])

    def test_reject_mislocalized_single_modification(self):
        with self.assertRaises(ValueError):
            single_structure('ACDK', '3,Carbamidomethyl[C];')

    def test_crosslink_beta_modification_offset(self):
        alpha, beta = 'AAKAA', 'ACKAA'
        sequence = alpha + '(3)-' + beta + '(3)'
        position = len(alpha) + 3 + 2
        smiles_a, fragments_a = crosslink_structure(sequence, 'Carbamidomethyl[C]({})'.format(position), 'DSSO')
        smiles_b, fragments_b = crosslink_structure(sequence, '{},Carbamidomethyl[C];'.format(position), 'DSSO')
        self.assertEqual(smiles_a, smiles_b)
        self.assertTrue(torch.equal(composition(fragments_a), composition(fragments_b)))
        with self.assertRaises(ValueError):
            crosslink_structure(sequence, 'Carbamidomethyl[C]({})'.format(position - 1), 'DSSO')

    def test_mgf_header_order_and_fragment_channels(self):
        text = ('BEGIN IONS\nNCE=25\nINSTRUMENT_NAME=Lumos\n'
                'SEQ=ACDK@2,Carbamidomethyl[C];@2@25\nCHARGE=2+\n'
                'ACTIVATION_TYPE=HCD\nINSTRUMENT_TYPE=FT\nTITLE=example\n'
                '100 10 b1-z1\n200 20 y2-z2\n250 40 Unknown\nEND IONS\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.mgf'
            path.write_text(text)
            record = read_records(path, 'ms2')[0]
        self.assertEqual(tuple(record['targets'].shape), (3, 8))
        self.assertAlmostEqual(record['targets'][0, 0].item(), 0.25)
        self.assertAlmostEqual(record['targets'][1, 3].item(), 0.5)
        self.assertEqual(record['charge'], 2)

    def test_peptide_identity_blocks_modified_variant_leakage(self):
        rows = {'train': [{'seq': 'ACDK', 'mod': ''}],
                'val': [{'seq': 'ACDK', 'mod': '2,Carbamidomethyl[C];'}]}
        with self.assertRaises(ValueError):
            check_splits(rows, 'single_rt')

    def test_crosslink_swapped_chains_are_same_group(self):
        rows = {'train': [{'seq': 'AAK(3)-CCK(3)', 'linker': 'DSSO'}],
                'test': [{'seq': 'CCK(3)-AAK(3)', 'linker': 'DSSO'}]}
        with self.assertRaises(ValueError):
            check_splits(rows, 'crosslink_rt')

    def test_ms2_batches_have_equal_lengths_and_complete_coverage(self):
        rows = [{'seq': sequence} for sequence in ['AA', 'AAA', 'AAAA', 'CC', 'CCC', 'CCCC', 'DD']]
        sampler = LengthBatchSampler(rows, 'ms2', batch_size=2, shuffle=True, seed=7)
        for _ in range(2):
            batches = list(sampler)
            self.assertEqual(sorted(index for batch in batches for index in batch), list(range(len(rows))))
            for batch in batches:
                self.assertEqual(len({len(rows[index]['seq']) for index in batch}), 1)


if __name__ == '__main__':
    unittest.main()
