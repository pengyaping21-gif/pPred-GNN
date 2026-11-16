"""Strict, portable checkpoint loading with architecture and RT-scale metadata."""
import re
from pathlib import Path

import torch


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Expected a PyTorch state dictionary or pPred checkpoint')
    state = payload.get('state_dict', payload.get('model_state_dict', payload))
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError('Checkpoint does not contain a tensor state dictionary')
    if all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    metadata = payload if 'state_dict' in payload else {}
    return state, metadata


def architecture(state, task):
    if 'gnn.x_embedding1.weight' not in state:
        raise ValueError('Checkpoint is not the full pPred GNN model; ablation/backbone-only weights are unsupported')
    layers = [int(match[1]) for key in state
              for match in [re.match(r'gnn\.gnns\.(\d+)\.', key)] if match]
    config = {
        'num_layer': max(layers) + 1,
        'emb_dim': state['gnn.x_embedding1.weight'].shape[1],
        'JK': 'last', 'graph_pooling': 'attention' if any(key.startswith('pool.gate_nn.') for key in state) else 'mean',
        'gnn_type': 'gin', 'drop_ratio': 0.3 if task == 'ms2' else 0.5,
    }
    is_ms2 = any(key.startswith('hidden_nn.bert.') for key in state)
    if is_ms2 != (task == 'ms2'):
        raise ValueError('An RT checkpoint cannot initialize MS2, and an MS2 checkpoint cannot initialize RT')
    if is_ms2:
        transformer_layers = [int(match[1]) for key in state
                              for match in [re.match(r'hidden_nn\.bert\.layer\.(\d+)\.', key)] if match]
        config.update({
            'hidden': state['hidden_nn.bert.layer.0.attention.self.query.weight'].shape[0] - 128,
            'nlayers': max(transformer_layers) + 1,
            'num_modloss_types': 4 if any(key.startswith('modloss_nn.') for key in state) else 0,
            'num_frag_types': 8, 'mask_modloss': False, 'dropout': 0.1,
        })
        output_channels = state['output_nn.nn.2.weight'].shape[0] + config['num_modloss_types']
        if output_channels != 8:
            raise ValueError('Expected the eight-channel b/y pPred MS2 checkpoint')
    return config


def build_model(task, device, checkpoint=None, allow_random_init=False):
    metadata = {}
    state = None
    if checkpoint:
        state, metadata = load_checkpoint(checkpoint)
        config = metadata.get('architecture') or architecture(state, task)
    elif allow_random_init:
        config = {'num_layer': 10, 'emb_dim': 64 if task == 'ms2' else 300,
                  'JK': 'last', 'graph_pooling': 'attention', 'gnn_type': 'gin',
                  'drop_ratio': 0.3 if task == 'ms2' else 0.5}
        if task == 'ms2':
            config.update(num_frag_types=8, num_modloss_types=4, mask_modloss=False, dropout=0.1)
    else:
        raise ValueError('Finetuning requires --checkpoint; --allow-random-init is only for an explicitly requested training smoke test')
    # Validate the task even when metadata supplied an architecture.
    if state is not None:
        architecture(state, task)
    if task == 'ms2':
        from .models.ms2 import ModelMS2Bert
        model = ModelMS2Bert(num_tasks=1, device=device, **config)
    else:
        from .models.rt import MolGraphPred
        model = MolGraphPred(num_tasks=1, device=device, **config)
    if state is not None:
        model.load_state_dict(state, strict=True)
    model.to(device)
    return model, config, metadata


def save_checkpoint(path, model, task, config, normalization, epoch, metrics, source_checkpoint):
    payload = {
        'format': 'ppred_finetune_v1',
        'task': task,
        'architecture': dict(config),
        'normalization': normalization,
        'epoch': int(epoch),
        'validation_metrics': {key: float(value) for key, value in metrics.items()},
        'initialization_checkpoint': Path(source_checkpoint).name if source_checkpoint else None,
        'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(payload, path)
