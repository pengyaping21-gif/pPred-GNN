"""Finetuning with validation-based checkpoint selection and held-out testing."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import build_model, save_checkpoint
from .data import check_splits, read_records
from .runtime import device_from_string, evaluate, fingerprint, make_loader, output_and_target, training_normalization


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finetune(args):
    if args.epochs < 1:
        raise ValueError('--epochs must be positive')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = device_from_string(args.device)
    splits = {'train': read_records(args.train, args.task), 'val': read_records(args.val, args.task)}
    if args.test:
        splits['test'] = read_records(args.test, args.task)
    split_summary = check_splits(splits, args.task)
    model, config, metadata = build_model(args.task, device, args.checkpoint, args.allow_random_init)
    normalization = training_normalization(args, splits['train'], metadata)
    if args.freeze_gnn:
        for parameter in model.gnn.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    loaders = {name: make_loader(records, args.task, normalization, args.batch_size,
                                name == 'train', args.seed) for name, records in splits.items()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / 'best.pth'
    if best_path.exists() and not args.overwrite:
        raise ValueError('Output already contains best.pth; choose a new output or pass --overwrite')
    manifest = {
        'task': args.task, 'seed': args.seed, 'device': str(device), 'normalization': normalization,
        'architecture': config, 'split_summary': split_summary,
        'training_kind': 'finetune' if args.checkpoint else 'random_initialization_smoke_test',
        'initialization': fingerprint(args.checkpoint) if args.checkpoint else None,
        'inputs': {name: fingerprint(path) for name, path in [('train', args.train), ('val', args.val), ('test', args.test)] if path},
        'hyperparameters': {'epochs': args.epochs, 'batch_size': args.batch_size,
                            'lr': args.lr, 'weight_decay': args.weight_decay, 'freeze_gnn': args.freeze_gnn,
                            'rt_normalization': args.rt_normalization},
    }
    (output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'device': str(device), 'splits': split_summary, 'normalization': normalization}), flush=True)
    best_loss, history = float('inf'), []
    started = time.monotonic()
    log_every = getattr(args, 'log_every', 25)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_gnn:
            model.gnn.eval()  # Do not update frozen batchnorm statistics.
        loss_sum, elements = 0., 0
        for step, batch in enumerate(loaders['train'], 1):
            optimizer.zero_grad(set_to_none=True)
            prediction, target = output_and_target(model, batch.to(device), args.task)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is non-finite')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += loss.item() * target.numel()
            elements += target.numel()
            if log_every and (step % log_every == 0 or step == len(loaders['train'])):
                progress = {'stage': 'training', 'epoch': epoch, 'epochs': args.epochs,
                            'step': step, 'steps': len(loaders['train']),
                            'train_mse': loss_sum / elements,
                            'elapsed_seconds': round(time.monotonic() - started, 1)}
                (output / 'progress.json').write_text(json.dumps(progress) + '\n')
                print(json.dumps(progress), flush=True)
        validation = evaluate(model, loaders['val'], args.task, normalization, device)
        row = {'epoch': epoch, 'train_mse': loss_sum / elements, 'val': validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        if validation['mse'] < best_loss:
            best_loss = validation['mse']
            save_checkpoint(best_path, model, args.task, config, normalization, epoch, validation, args.checkpoint)
        (output / 'progress.json').write_text(json.dumps({
            'stage': 'epoch_complete', 'epoch': epoch, 'epochs': args.epochs,
            'val': validation, 'elapsed_seconds': round(time.monotonic() - started, 1),
        }) + '\n')
    model, _, _ = build_model(args.task, device, best_path)
    metrics = {'val': evaluate(model, loaders['val'], args.task, normalization, device)}
    if 'test' in loaders:
        metrics['test'] = evaluate(model, loaders['test'], args.task, normalization, device)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (output / 'progress.json').write_text(json.dumps({
        'stage': 'complete', 'epochs': args.epochs,
        'elapsed_seconds': round(time.monotonic() - started, 1), 'metrics': metrics,
    }) + '\n')
    print(json.dumps({'checkpoint': str(best_path), 'metrics': metrics}), flush=True)


