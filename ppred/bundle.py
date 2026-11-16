"""Resolve Release checkpoints with a pinned checksum and a portable path."""
import json
from pathlib import Path

from .runtime import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def bundled_checkpoint(task, root=None):
    root = Path(root) if root is not None else ROOT
    manifest = root / 'checkpoints/manifest.json'
    if not manifest.is_file():
        raise ValueError('Bundled checkpoint manifest is missing; use --checkpoint PATH')
    entries = json.loads(manifest.read_text())['checkpoints']
    if task not in entries:
        raise ValueError('No bundled checkpoint for task ' + task)
    entry = entries[task]
    path = (root / entry['path']).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Checkpoint manifest path must stay inside the bundle')
    if not path.is_file():
        raise ValueError('Checkpoint is missing: ' + str(path) + '. Download and extract the checkpoint archive from '
                         'https://github.com/pengyaping21-gif/pPred-GNN/releases/tag/v0.2.0 '
                         'in the project root, or use --checkpoint PATH')
    actual = fingerprint(path)
    if actual['bytes'] != entry['bytes'] or actual['sha256'] != entry['sha256']:
        raise ValueError('Bundled checkpoint checksum mismatch: ' + str(path))
    return path
