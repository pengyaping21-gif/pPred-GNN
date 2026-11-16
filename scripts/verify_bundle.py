#!/usr/bin/env python3
"""Verify the three checkpoints, nine demo partitions and source-package location."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ppred
import torch
from ppred.bundle import bundled_checkpoint
from ppred.checkpoints import build_model
from ppred.chemistry import molecular_features
from ppred.data import check_splits, read_records
from ppred.runtime import fingerprint


def verify():
    torch.set_num_threads(2)
    if Path(ppred.__file__).resolve().parent != ROOT / 'ppred':
        raise ValueError('Imported a different ppred installation')
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text())
    provenance = json.loads((ROOT / 'demo/provenance.json').read_text())
    result = {'package_version': ppred.__version__, 'tasks': {}}
    for task, entry in manifest['checkpoints'].items():
        checkpoint = bundled_checkpoint(task)
        model, _, metadata = build_model(task, torch.device('cpu'), checkpoint)
        if metadata.get('task') != task or metadata.get('normalization') != entry['normalization']:
            raise ValueError('Checkpoint metadata differs from manifest: ' + task)
        splits = {}
        for split, partition in provenance['tasks'][task]['partitions'].items():
            expected = partition['demo']
            path = ROOT / 'demo' / expected['path']
            actual = fingerprint(path)
            if actual['sha256'] != expected['sha256'] or actual['bytes'] != expected['bytes']:
                raise ValueError('Demo checksum mismatch: ' + str(path))
            records = read_records(path, task)
            for row in records:
                molecular_features(task, row['seq'], row['mod'], row.get('linker', ''))
            splits[split] = records
        summary = check_splits(splits, task)
        if summary != provenance['tasks'][task]['splits']:
            raise ValueError('Demo record counts differ from provenance: ' + task)
        result['tasks'][task] = {'checkpoint_sha256': entry['sha256'], 'strict_model_load': True,
                                'demo_splits': summary, 'peptide_group_overlap': 0}
        del model
    result['status'] = 'passed'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
