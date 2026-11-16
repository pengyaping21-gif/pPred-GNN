#!/usr/bin/env python3
"""Sample real records without changing the source train/validation/test membership."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppred.chemistry import molecular_features
from ppred.data import check_splits, group_key, read_records
from ppred.runtime import fingerprint

TASKS = ['single_rt', 'crosslink_rt', 'ms2']
COUNTS = {'train': 32, 'val': 8, 'test': 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, required=True, help='Directory with <task>/{train,val,test}.{rtdata,mgf}')
    parser.add_argument('--output', type=Path, required=True, help='A new or empty demo directory')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be new or empty')
    report = {'seed': args.seed, 'selection': 'one record per peptide group, sampled within each fixed source split',
              'tasks': {}}
    for task in TASKS:
        extension = '.mgf' if task == 'ms2' else '.rtdata'
        chosen, provenance = {}, {}
        directory = args.output / task
        directory.mkdir(parents=True, exist_ok=True)
        for offset, (split, count) in enumerate(COUNTS.items()):
            source = args.splits / task / (split + extension)
            records = read_records(source, task)
            indices = list(range(len(records)))
            random.Random(args.seed + offset).shuffle(indices)
            selected, selected_indices, groups = [], [], set()
            for index in indices:
                record = records[index]
                key = group_key(record, task)
                if key in groups:
                    continue
                # Fixed source splits were already chemically filtered; fail loudly if they changed.
                molecular_features(task, record['seq'], record['mod'], record.get('linker', ''))
                groups.add(key)
                selected.append(record)
                selected_indices.append(index)
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError('Not enough peptide groups in ' + str(source))
            destination = directory / (split + extension)
            if task == 'ms2':
                destination.write_text(''.join(row['mgf_block'] for row in selected))
            else:
                fields = ['seq', 'mod'] + (['linker'] if task == 'crosslink_rt' else []) + ['rt']
                with destination.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fields, delimiter='\t', lineterminator='\n')
                    writer.writeheader()
                    writer.writerows({key: row[key] for key in fields} for row in selected)
            chosen[split] = selected
            provenance[split] = {'source': dict(fingerprint(source), path=task + '/' + source.name),
                                 'source_records': len(records), 'selected_source_indices_zero_based': selected_indices,
                                 'demo': dict(fingerprint(destination), path=task + '/' + destination.name)}
        report['tasks'][task] = {'splits': check_splits(chosen, task), 'partitions': provenance}
    (args.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({task: value['splits'] for task, value in report['tasks'].items()}, indent=2))


if __name__ == '__main__':
    main()
