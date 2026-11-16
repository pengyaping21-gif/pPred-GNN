#!/usr/bin/env python3
"""Create training partitions with no peptide-backbone overlap."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rdkit import Chem
from ppred.chemistry import crosslink_structure, single_structure
from ppred.data import check_splits, group_key, mgf_blocks, read_records, spectrum_record
from ppred.runtime import fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['single_rt', 'crosslink_rt', 'ms2'])
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--skip-invalid', action='store_true', help='Exclude unsupported records and save counts/examples')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be a new or empty directory')
    records, excluded, duplicates, seen = [], [], 0, set()
    for path in args.input:
        if args.task == 'ms2':
            source_records = enumerate(mgf_blocks(path))
        else:
            source_records = enumerate(read_records(path, args.task))
        for index, item in source_records:
            try:
                record = spectrum_record(item, index) if args.task == 'ms2' else item
                if args.task == 'crosslink_rt':
                    smiles, _ = crosslink_structure(record['seq'], record['mod'], record['linker'])
                else:
                    smiles, _ = single_structure(record['seq'], record['mod'])
                if Chem.MolFromSmiles(smiles) is None:
                    raise ValueError('Invalid molecular graph')
            except (ValueError, KeyError) as error:
                if not args.skip_invalid:
                    raise
                excluded.append({'source': path.name, 'row_or_spectrum': index, 'reason': str(error)})
                continue
            identity = record['mgf_block'] if args.task == 'ms2' else json.dumps(
                [record['seq'], record['mod'], record['linker'], record['rt']])
            key = hashlib.sha256(identity.encode()).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            records.append(record)
    grouped = defaultdict(list)
    for record in records:
        grouped[group_key(record, args.task)].append(record)
    keys = sorted(grouped, key=repr)
    if len(keys) < 3:
        raise ValueError('At least three distinct peptide groups are required')
    random.Random(args.seed).shuffle(keys)
    training_count = max(1, int(len(keys) * .8))
    validation_count = max(1, int(len(keys) * .1))
    if training_count + validation_count >= len(keys):
        training_count, validation_count = len(keys) - 2, 1
    partitions = {'train': keys[:training_count],
                  'val': keys[training_count:training_count + validation_count],
                  'test': keys[training_count + validation_count:]}
    splits = {name: [record for key in group_ids for record in grouped[key]]
              for name, group_ids in partitions.items()}
    summary = check_splits(splits, args.task)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        if args.task == 'ms2':
            (args.output / (name + '.mgf')).write_text(''.join(row['mgf_block'] for row in rows))
        else:
            fields = ['seq', 'mod'] + (['linker'] if args.task == 'crosslink_rt' else []) + ['rt']
            with (args.output / (name + '.rtdata')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
                writer.writeheader()
                writer.writerows({key: record[key] for key in fields} for record in rows)
    report = {'task': args.task, 'seed': args.seed, 'group_fractions': [.8, .1, .1],
              'sources': [fingerprint(path) for path in args.input], 'splits': summary,
              'exact_duplicates_removed': duplicates, 'excluded_records': len(excluded),
              'exclusion_reasons': dict(Counter(item['reason'] for item in excluded)),
              'excluded_examples': excluded[:20]}
    (args.output / 'split_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
