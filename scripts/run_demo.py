#!/usr/bin/env python3
"""Run baseline evaluation, real finetuning, held-out evaluation and prediction."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppred.bundle import bundled_checkpoint
from ppred.runtime import fingerprint


def run(command, log):
    print(' '.join(str(part) for part in command), flush=True)
    with log.open('w') as stream:
        result = subprocess.run([str(part) for part in command], cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Command failed; inspect ' + str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['all', 'single_rt', 'crosslink_rt', 'ms2'], default='all')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=1, help='Demo default is one epoch, not a performance benchmark')
    parser.add_argument('--output', type=Path, help='Must be a new directory; default outputs/demo_<timestamp>')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    output = args.output or ROOT / 'outputs' / ('demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output = output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    tasks = ['single_rt', 'crosslink_rt', 'ms2'] if args.task == 'all' else [args.task]
    # Check all requested inputs before starting a partially valid run.
    configs = {}
    for task in tasks:
        config = json.loads((ROOT / 'configs' / (task + '.json')).read_text())
        if config['task'] != task:
            raise ValueError('Configuration task mismatch: ' + task)
        config['checkpoint'] = str(bundled_checkpoint(task))
        for split in ['train', 'val', 'test']:
            config[split] = str(ROOT / config[split])
            if not Path(config[split]).is_file():
                raise ValueError('Missing data: ' + config[split])
        configs[task] = config
    output.mkdir(parents=True)
    summary = {'scope': 'small real-data pipeline verification; adapted initialization, not a new benchmark',
               'epochs': args.epochs, 'tasks': {}}
    (output / 'status.json').write_text(json.dumps({'status': 'running', 'tasks': tasks}) + '\n')
    try:
        for task in tasks:
            config = configs[task]
            directory = output / task
            directory.mkdir()
            (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
            common = [sys.executable, '-m', 'ppred']
            runtime = ['--task', task, '--device', args.device, '--batch-size', config['batch_size'],
                       '--cpu-threads', config['cpu_threads']]
            run(common + ['evaluate'] + runtime + ['--checkpoint', config['checkpoint'], '--input', config['test'],
                                                   '--output', directory / 'before.json'], directory / 'before.log')
            training = directory / 'finetune'
            run(common + ['finetune'] + runtime + ['--checkpoint', config['checkpoint'], '--train', config['train'],
                '--val', config['val'], '--test', config['test'], '--epochs', args.epochs,
                '--lr', config['lr'], '--weight-decay', config['weight_decay'], '--seed', config['seed'],
                '--rt-normalization', config['rt_normalization'], '--output', training], directory / 'finetune.log')
            best = training / 'best.pth'
            run(common + ['evaluate'] + runtime + ['--checkpoint', best, '--input', config['test'],
                                                   '--output', directory / 'after.json'], directory / 'after.log')
            prediction = directory / ('predictions.jsonl' if task == 'ms2' else 'predictions.csv')
            run(common + ['predict'] + runtime + ['--checkpoint', best, '--input', config['test'],
                                                  '--output', prediction], directory / 'predict.log')
            before = json.loads((directory / 'before.json').read_text())
            after = json.loads((directory / 'after.json').read_text())
            if fingerprint(best)['sha256'] == fingerprint(config['checkpoint'])['sha256']:
                raise ValueError('Finetuning did not create a different checkpoint')
            summary['tasks'][task] = {'before': before['metrics'], 'after': after['metrics'],
                                     'test_records': after['records'], 'checkpoint': str(best.relative_to(output)),
                                     'predictions': str(prediction.relative_to(output))}
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps({'task': task, 'result': summary['tasks'][task]}), flush=True)
    except Exception as error:
        (output / 'status.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}) + '\n')
        raise
    (output / 'status.json').write_text(json.dumps({'status': 'complete', 'tasks': tasks}) + '\n')
    print('Completed: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
