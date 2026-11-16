"""Prediction and standalone labelled-data evaluation."""
import csv
import json
from pathlib import Path

import torch

from .checkpoints import build_model
from .data import FRAGMENT_TYPES, read_records
from .runtime import device_from_string, evaluate, fingerprint, inference_normalization, make_loader


def load_inputs(args, require_labels):
    output = Path(args.output)
    if output.exists() and not getattr(args, 'overwrite', False):
        raise ValueError('Output already exists; choose a new path or pass --overwrite')
    torch.set_num_threads(args.cpu_threads)
    device = device_from_string(args.device)
    model, _, metadata = build_model(args.task, device, args.checkpoint)
    normalization = inference_normalization(args, metadata)
    records = read_records(args.input, args.task, require_labels=require_labels)
    loader = make_loader(records, args.task, normalization, args.batch_size)
    return output, model, device, normalization, records, loader


def evaluate_file(args):
    output, model, device, normalization, records, loader = load_inputs(args, require_labels=True)
    result = {'task': args.task, 'records': len(records), 'device': str(device),
              'checkpoint': fingerprint(args.checkpoint), 'input': fingerprint(args.input),
              'normalization': normalization,
              'metrics': evaluate(model, loader, args.task, normalization, device)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def predict(args):
    output, model, device, normalization, records, loader = load_inputs(args, require_labels=False)
    predictions = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            indices = batch.record_id.tolist()
            batch = batch.to(device)
            prediction = model(batch) if args.task == 'ms2' else model(batch)[0].reshape(-1)
            if args.task != 'ms2':
                prediction = prediction * (normalization['maximum'] - normalization['minimum']) + normalization['minimum']
            if not torch.isfinite(prediction).all():
                raise ValueError('Prediction produced non-finite values')
            for index, value in zip(indices, prediction.cpu()):
                predictions[index] = value.clamp_min(0).tolist() if args.task == 'ms2' else value.item()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        if args.task == 'ms2':
            for index, record in enumerate(records):
                row = {'id': record['id'], 'seq': record['seq'], 'mod': record['mod'],
                       'charge': record['charge'], 'nce': record['nce'],
                       'instrument': record['instrument'], 'activation': record['activation'],
                       'fragment_types': FRAGMENT_TYPES, 'intensities': predictions[index]}
                stream.write(json.dumps(row) + '\n')
        else:
            writer = csv.DictWriter(stream, fieldnames=['id', 'seq', 'mod', 'linker', 'observed_rt', 'predicted_rt', 'rt_unit'])
            writer.writeheader()
            for index, record in enumerate(records):
                writer.writerow({'id': record['id'], 'seq': record['seq'], 'mod': record['mod'],
                                 'linker': record['linker'], 'observed_rt': record['rt'],
                                 'predicted_rt': predictions[index], 'rt_unit': normalization.get('unit', 'source_units')})
    print(json.dumps({'records': len(records), 'output': str(output)}))
