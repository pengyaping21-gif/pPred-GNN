import argparse


TASKS = ['single_rt', 'crosslink_rt', 'ms2']


def main():
    parser = argparse.ArgumentParser(description='pPred: finetune, evaluate and predict')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('finetune', help='Finetune; select the best epoch on validation MSE')
    prediction = commands.add_parser('predict', help='Predict RT or eight-channel fragment intensities')
    evaluation = commands.add_parser('evaluate', help='Evaluate a checkpoint on labelled data')
    for command in (train, prediction, evaluation):
        command.add_argument('--task', required=True, choices=TASKS)
        command.add_argument('--checkpoint', help='Defaults to the verified bundled checkpoint for this task')
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='auto', help='auto, cpu or cuda:0')
        command.add_argument('--batch-size', type=int, default=8)
        command.add_argument('--cpu-threads', type=int, default=2)
        command.add_argument('--rt-min', type=float)
        command.add_argument('--rt-max', type=float)
        command.add_argument('--rt-unit', default='source_units')
        command.add_argument('--overwrite', action='store_true')
    train.add_argument('--train', required=True)
    train.add_argument('--val', required=True)
    train.add_argument('--test')
    train.add_argument('--epochs', type=int, default=10)
    train.add_argument('--lr', type=float, default=1e-4)
    train.add_argument('--weight-decay', type=float, default=0.01)
    train.add_argument('--seed', type=int, default=2024)
    train.add_argument('--freeze-gnn', action='store_true')
    train.add_argument('--allow-random-init', action='store_true', help='Explicit training smoke test; does not count as pretrained finetuning')
    train.add_argument('--rt-normalization', choices=['auto', 'train', 'checkpoint'], default='auto',
                       help='auto keeps a same-task checkpoint scale; train fits only the new train split')
    train.add_argument('--log-every', type=int, default=25, help='Log training progress every N batches; 0 disables batch logs')
    for command in (prediction, evaluation):
        command.add_argument('--input', required=True)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.batch_size < 1:
        parser.error('CPU threads and batch size must be positive')
    if args.command == 'finetune' and (args.log_every < 0 or args.lr <= 0 or args.weight_decay < 0):
        parser.error('Learning rate must be positive; log frequency and weight decay must be nonnegative')
    if getattr(args, 'allow_random_init', False) and args.checkpoint:
        parser.error('--allow-random-init and --checkpoint are mutually exclusive')
    try:
        if not args.checkpoint and not getattr(args, 'allow_random_init', False):
            from .bundle import bundled_checkpoint
            args.checkpoint = str(bundled_checkpoint(args.task))
        if args.command == 'finetune':
            from .training import finetune
            finetune(args)
        else:
            from .inference import evaluate_file, predict
            (evaluate_file if args.command == 'evaluate' else predict)(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, 'Error: ' + str(error) + '\n')
