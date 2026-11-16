"""Shared batching, evaluation and runtime helpers."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .data import LengthBatchSampler, PeptideDataset


def device_from_string(value):
    if value == 'auto':
        value = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this process; use the server terminal or --device cpu')
    return device


def make_loader(records, task, normalization, batch_size, shuffle=False, seed=2024):
    dataset = PeptideDataset(records, task, normalization)
    sampler = LengthBatchSampler(records, task, batch_size, shuffle, seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def output_and_target(model, batch, task):
    if task == 'ms2':
        prediction = model(batch)
        target = batch.exp_spec.reshape(prediction.shape)
    else:
        prediction = model(batch)[0].reshape(-1)
        target = batch.y.reshape(-1)
    return prediction, target


def evaluate(model, loader, task, normalization, device):
    model.eval()
    squared_error, elements, absolute_error, count = 0., 0, 0., 0
    cosines = []
    with torch.no_grad():
        for batch in loader:
            prediction, target = output_and_target(model, batch.to(device), task)
            squared_error += (prediction - target).square().sum().item()
            elements += target.numel()
            if task == 'ms2':
                prediction = prediction.clamp_min(0).flatten(1)
                target = target.flatten(1)
                cosines.extend(torch.nn.functional.cosine_similarity(prediction, target, dim=1).cpu().tolist())
            else:
                scale = normalization['maximum'] - normalization['minimum']
                absolute_error += (prediction - target).abs().sum().item() * scale
                count += target.numel()
    if elements == 0:
        raise ValueError('Cannot evaluate an empty dataset')
    metrics = {'mse': squared_error / elements}
    if task == 'ms2':
        metrics['mean_cosine'] = float(np.mean(cosines))
    else:
        metrics['mae'] = absolute_error / max(count, 1)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError('Evaluation produced non-finite metrics')
    return metrics


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'name': Path(path).name, 'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def validate_normalization(normalization):
    if not isinstance(normalization, dict):
        raise ValueError('RT normalization metadata is missing')
    minimum, maximum = normalization.get('minimum'), normalization.get('maximum')
    if minimum is None or maximum is None or not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError('RT normalization requires finite minimum < maximum')
    return dict(normalization)


def training_normalization(args, records, metadata):
    if args.task == 'ms2':
        return None
    if (args.rt_min is None) != (args.rt_max is None):
        raise ValueError('Specify both --rt-min and --rt-max, or neither')
    if args.rt_min is not None:
        return validate_normalization({'minimum': args.rt_min, 'maximum': args.rt_max,
                                       'fit_split': 'explicit', 'unit': args.rt_unit})
    policy = getattr(args, 'rt_normalization', 'auto')
    compatible = metadata.get('task') == args.task and metadata.get('normalization') is not None
    if policy == 'checkpoint' or (policy == 'auto' and compatible):
        if not compatible:
            raise ValueError('--rt-normalization checkpoint requires a checkpoint for the same task with RT scale metadata')
        normalization = validate_normalization(metadata['normalization'])
        normalization['origin'] = 'initialization_checkpoint'
        return normalization
    values = [record['rt'] for record in records]
    return validate_normalization({'minimum': min(values), 'maximum': max(values),
                                   'fit_split': 'train', 'unit': args.rt_unit,
                                   'origin': 'current_training_split'})


def inference_normalization(args, metadata):
    if metadata.get('task') and metadata['task'] != args.task:
        raise ValueError('Input task differs from checkpoint task; finetune to the new task first')
    if args.task == 'ms2':
        return None
    if metadata.get('normalization') is not None:
        normalization = validate_normalization(metadata['normalization'])
        if args.rt_min is not None or args.rt_max is not None:
            if args.rt_min != normalization['minimum'] or args.rt_max != normalization['maximum']:
                raise ValueError('RT scale arguments differ from the scale embedded in this checkpoint')
        return normalization
    if args.rt_min is None or args.rt_max is None:
        raise ValueError('Raw RT state_dict lacks its scale; supply the source --rt-min and --rt-max')
    return validate_normalization({'minimum': args.rt_min, 'maximum': args.rt_max, 'unit': args.rt_unit})


