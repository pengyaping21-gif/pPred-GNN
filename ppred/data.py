"""Validated RT tables, annotated MGF spectra and peptide-group-aware batches."""
import csv
import math
import random
import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler

from .chemistry import molecular_features, parse_crosslink, validate_sequence

FRAGMENT_TYPES = ['b-z1', 'b-z2', 'y-z1', 'y-z2', 'b-ModLoss-z1',
                  'b-ModLoss-z2', 'y-ModLoss-z1', 'y-ModLoss-z2']
INSTRUMENTS = ['QE', 'Velos', 'Elite', 'Fusion', 'Lumos']
ACTIVATIONS = ['HCD-FT', 'HCD-IT', 'CID-FT', 'CID-IT']


def read_rt(path, task, require_labels=True):
    records = []
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        required = {'seq', 'mod'} | ({'rt'} if require_labels else set())
        if task == 'crosslink_rt':
            required.add('linker')
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(str(path) + ' must contain tab-separated columns: ' + ', '.join(sorted(required)))
        for line_number, row in enumerate(reader, 2):
            sequence = row['seq'].strip().upper()
            validate_sequence(sequence) if task == 'single_rt' else parse_crosslink(sequence)
            label = row.get('rt', '').strip()
            rt = float(label) if label else None
            if (rt is not None and not math.isfinite(rt)) or (rt is None and require_labels):
                raise ValueError('Invalid RT on line ' + str(line_number))
            records.append({'seq': sequence, 'mod': (row.get('mod') or '').strip(),
                            'linker': (row.get('linker') or '').strip(), 'rt': rt,
                            'id': str(line_number - 2)})
    if not records:
        raise ValueError('Empty input table: ' + str(path))
    return records


def mgf_blocks(path):
    current = None
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.upper() == 'BEGIN IONS':
                if current is not None:
                    raise ValueError('Nested BEGIN IONS on line ' + str(line_number))
                current = {'metadata': {}, 'peaks': [], 'lines': ['BEGIN IONS']}
            elif line.upper() == 'END IONS':
                if current is None:
                    raise ValueError('END IONS without a spectrum')
                current['lines'].append('END IONS')
                yield current
                current = None
            elif current is not None:
                current['lines'].append(line)
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.upper()
                    if key in current['metadata']:
                        raise ValueError('Duplicated MGF metadata key: ' + key)
                    current['metadata'][key] = value
                else:
                    columns = line.split()
                    if len(columns) not in (2, 3):
                        raise ValueError('Expected m/z intensity [annotation] on line ' + str(line_number))
                    mz, intensity = float(columns[0]), float(columns[1])
                    if not math.isfinite(mz) or not math.isfinite(intensity) or mz <= 0 or intensity < 0:
                        raise ValueError('Invalid peak on line ' + str(line_number))
                    current['peaks'].append((mz, intensity, columns[2] if len(columns) == 3 else 'Unknown'))
    if current is not None:
        raise ValueError('Unterminated spectrum at the end of ' + str(path))


def spectrum_record(block, index, require_labels=True):
    meta = block['metadata']
    encoded = meta.get('SEQ', meta.get('SEQUENCE', ''))
    fields = encoded.split('@')
    sequence = fields[0].upper()
    validate_sequence(sequence)
    if not 2 <= len(sequence) <= 60:
        raise ValueError('MS2 requires peptide lengths from 2 through 60')
    modifications = fields[1] if len(fields) > 1 else ''
    charge_text = meta.get('CHARGE', fields[2] if len(fields) > 2 else '').rstrip('+')
    nce_text = meta.get('NCE', fields[3] if len(fields) > 3 else '')
    if not charge_text or not nce_text:
        raise ValueError('Every spectrum needs CHARGE and NCE')
    charge, nce = int(charge_text), float(nce_text)
    if not 1 <= charge <= 10 or not math.isfinite(nce) or not 0 <= nce <= 100:
        raise ValueError('CHARGE must be 1–10 and NCE must be 0–100')
    instrument = meta.get('INSTRUMENT_NAME', '')
    instrument = next((item for item in INSTRUMENTS if item.lower() == instrument.lower()), instrument)
    activation = meta.get('ACTIVATION_TYPE', '').upper() + '-' + meta.get('INSTRUMENT_TYPE', '').upper()
    if instrument not in INSTRUMENTS or activation not in ACTIVATIONS:
        raise ValueError('Unsupported instrument or activation: ' + instrument + ', ' + activation)
    targets = torch.zeros((len(sequence) - 1, len(FRAGMENT_TYPES)), dtype=torch.float32)
    peak_max = max((peak[1] for peak in block['peaks']), default=0.)
    annotation_count = 0
    for _, intensity, label in block['peaks']:
        match = re.fullmatch(r'([by])(\d+)-(ModLoss-)?z([12])', label)
        if not match:
            continue
        ion, number, modloss, z = match.groups()
        number = int(number)
        if not 1 <= number < len(sequence):
            raise ValueError('Fragment annotation exceeds its peptide length: ' + label)
        channel = ion + '-' + (modloss or '') + 'z' + z
        # The released source code normalizes against all peaks, including Unknown.
        targets[number - 1, FRAGMENT_TYPES.index(channel)] = intensity / peak_max if peak_max else 0.
        annotation_count += 1
    if require_labels and (annotation_count == 0 or targets.max().item() <= 0):
        raise ValueError('MS2 finetuning requires positive, annotated b/y fragment peaks')
    return {'id': meta.get('TITLE', str(index)), 'seq': sequence, 'mod': modifications,
            'charge': charge, 'nce': nce, 'instrument': instrument, 'activation': activation,
            'targets': targets, 'rt_seconds': float(meta['RTINSECONDS']) if 'RTINSECONDS' in meta else None,
            'mgf_block': '\n'.join(block['lines']) + '\n'}


def read_records(path, task, require_labels=True):
    if task != 'ms2':
        return read_rt(path, task, require_labels)
    records = [spectrum_record(block, i, require_labels) for i, block in enumerate(mgf_blocks(path))]
    if not records:
        raise ValueError('No spectra in ' + str(path))
    return records


def group_key(record, task):
    """Keep modified/charged variants of a peptide in one split."""
    if task == 'crosslink_rt':
        alpha, pos_a, beta, pos_b = parse_crosslink(record['seq'])
        pair = sorted([(alpha, pos_a), (beta, pos_b)])
        return tuple(pair) + (record['linker'],)
    return record['seq']


def check_splits(splits, task):
    groups = {name: {group_key(record, task) for record in records} for name, records in splits.items()}
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            shared = groups[left] & groups[right]
            if shared:
                raise ValueError('{} and {} share {} peptide groups; re-split by peptide identity'.format(left, right, len(shared)))
    return {name: {'records': len(splits[name]), 'peptide_groups': len(groups[name])} for name in names}


class PeptideDataset(Dataset):
    def __init__(self, records, task, normalization=None):
        self.records, self.task, self.normalization = records, task, normalization

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data = molecular_features(self.task, record['seq'], record['mod'], record.get('linker', '')).clone()
        # PyG increments attributes containing "index" by the number of nodes.
        data.record_id = torch.tensor([index])
        if self.task == 'ms2':
            data.nce = torch.tensor([record['nce'] / 100.], dtype=torch.float32)
            data.charge = torch.tensor([record['charge'] / 10.], dtype=torch.float32)
            data.inst_indices = torch.tensor([INSTRUMENTS.index(record['instrument'])])
            # Preserve the misspelled source field for checkpoint compatibility.
            data.actication = torch.tensor([ACTIVATIONS.index(record['activation'])], dtype=torch.float32)
            data.exp_spec = record['targets']
        elif record['rt'] is not None:
            minimum, maximum = self.normalization['minimum'], self.normalization['maximum']
            data.y = torch.tensor([(record['rt'] - minimum) / (maximum - minimum)], dtype=torch.float32)
        return data


class LengthBatchSampler(Sampler):
    """MS2 has no padding mask in the original network: batch equal lengths only."""
    def __init__(self, records, task, batch_size, shuffle=False, seed=2024):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.buckets = defaultdict(list)
        for index, record in enumerate(records):
            self.buckets[len(record['seq']) if task == 'ms2' else 0].append(index)
        self.batch_size = batch_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for source_indices in self.buckets.values():
            indices = list(source_indices)
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(indices[offset:offset + self.batch_size]
                           for offset in range(0, len(indices), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.buckets.values())
