"""Checkpoint-compatible pPred peptide graph, fingerprint and composition features."""
import csv
import os
import re
from functools import lru_cache
from pathlib import Path

import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from .fingerprints import GetPubChemFPs

RESOURCE_DIR = Path(__file__).parent / "resources"
AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_ELEMENTS = ['C', 'H', 'O', 'N', 'P', 'S', 'Na', 'Mg', 'K', 'Ca',
                    'Fe', 'Zn', 'Cu', 'Mn', 'Se', 'V', 'B', 'I']
ELEMENTS = SPECIAL_ELEMENTS + [
    Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 119)
    if Chem.GetPeriodicTable().GetElementSymbol(i) not in SPECIAL_ELEMENTS
]
ELEMENT_INDEX = {symbol: index for index, symbol in enumerate(ELEMENTS)}


@lru_cache(maxsize=1)
def dictionaries():
    def read(name):
        with (RESOURCE_DIR / name).open() as stream:
            return list(csv.DictReader(stream, delimiter=" "))
    amino = {r['amino_acid']: r for r in read('aa.ini')}
    modifications = {r['modification']: r for r in read('modification_smiles_new.ini')}
    linkers = {
        (r['linker_name'], r['alpha_site'], r['beta_site']): r['SMILES']
        for r in read('linker.ini')
    }
    return amino, modifications, linkers


def validate_sequence(sequence):
    if not sequence or any(aa not in AA for aa in sequence):
        raise ValueError("Expected a nonempty sequence of the 20 standard amino acids: " + sequence)


def single_modifications(text, length):
    """Original pPred format: 0,Acetyl[ProteinN-term];3,Oxidation[M];"""
    modifications = {}
    for token in (text or "").split(';'):
        if not token.strip():
            continue
        position, name = token.strip().split(',', 1)
        position = int(position)
        if not 0 <= position <= length + 1:
            raise ValueError("Modification position is outside the peptide: " + token)
        if position in modifications:
            raise ValueError("Multiple modifications at one position are unsupported: " + token)
        if name not in dictionaries()[1]:
            raise ValueError("Modification is missing from the supplied SMILES dictionary: " + name)
        modifications[position] = name
    return modifications


def residue_fragments(sequence, modifications):
    amino, mod_dict, _ = dictionaries()
    fragments = []
    for position, aa in enumerate(sequence, 1):
        name = modifications.get(position)
        fragments.append(mod_dict[name]['SMILES-Front'] if name else amino[aa]['SMILES-Front'])
    fragments[-1] += 'O'
    if 0 in modifications:
        fragments[0] = mod_dict[modifications[0]]['SMILES-Front'] + fragments[0]
    if len(sequence) + 1 in modifications:
        fragments[-1] = fragments[-1][:-1] + mod_dict[modifications[len(sequence) + 1]]['SMILES-Front']
    return fragments


def single_structure(sequence, modifications):
    validate_sequence(sequence)
    if len(sequence) > 60:
        raise ValueError("The source pPred RT/MS2 models support at most 60 residues")
    parsed = single_modifications(modifications, len(sequence))
    for position, name in parsed.items():
        expected = re.search(r'\[([A-Z])\]', name)
        if expected and (not 1 <= position <= len(sequence) or sequence[position - 1] != expected[1]):
            raise ValueError("Modification does not match its residue: " + str(position) + ',' + name)
    fragments = residue_fragments(sequence, parsed)
    return ''.join(fragments), fragments


def parse_crosslink(sequence):
    match = re.fullmatch(r'([A-Z]+)\((\d+)\)-([A-Z]+)\((\d+)\)', sequence)
    if not match:
        raise ValueError("Expected ALPHA(link_position)-BETA(link_position): " + sequence)
    alpha, pos_a, beta, pos_b = match.groups()
    validate_sequence(alpha)
    validate_sequence(beta)
    pos_a, pos_b = int(pos_a), int(pos_b)
    if not 1 <= pos_a <= len(alpha) or not 1 <= pos_b <= len(beta):
        raise ValueError("Crosslink position is outside its peptide: " + sequence)
    return alpha, pos_a, beta, pos_b


def crosslink_structure(sequence, modification_text, linker):
    alpha, pos_a, beta, pos_b = parse_crosslink(sequence)
    if len(alpha) + len(beta) + 1 > 60:
        raise ValueError("Crosslink composition exceeds 60 positions; filter these rows explicitly")
    amino, mod_dict, linkers = dictionaries()
    key = (linker, alpha[pos_a - 1], beta[pos_b - 1])
    if key not in linkers:
        raise ValueError("Unsupported linker/site combination: " + str(key))
    mods_a, mods_b = {}, {}
    for token in (modification_text or '').split(';'):
        if not token.strip():
            continue
        if ',' in token:
            position_text, name = token.strip().split(',', 1)
            position = int(position_text)
        else:
            match = re.fullmatch(r'(.+)\((\d+)\)', token.strip())
            if not match:
                raise ValueError("Expected crosslink modification Name(position) or position,Name: " + token)
            name, position = match[1], int(match[2])
        if name not in mod_dict:
            raise ValueError("Unknown modification: " + name)
        if position <= len(alpha) + 1:
            target, relative = mods_a, position
        else:
            # pLink absolute positions include alpha C-term, separator and beta N-term.
            # All residue-specific modifications in both supplied files match offset 3.
            target, relative = mods_b, position - len(alpha) - 3
        limit = len(alpha) if target is mods_a else len(beta)
        if not 0 <= relative <= limit + 1 or relative in target:
            raise ValueError("Invalid or duplicated crosslink modification: " + token)
        expected = re.search(r'\[([A-Z])\]', name)
        peptide = alpha if target is mods_a else beta
        if expected and (not 1 <= relative <= limit or peptide[relative - 1] != expected[1]):
            raise ValueError("Crosslink modification does not match its residue: " + token)
        target[relative] = name
    if pos_a in mods_a or pos_b in mods_b:
        raise ValueError("Modification at a crosslinked residue is unsupported")
    if len(alpha) + 1 in mods_a or len(beta) + 1 in mods_b:
        raise ValueError("C-terminal crosslink modifications are not supported by the source linker SMILES")

    def fragment(aa, position, mods, reverse=False):
        name = mods.get(position)
        field = 'SMILES-Reverse' if reverse else 'SMILES-Front'
        return mod_dict[name][field] if name else amino[aa][field]

    # Preserve the original pPred traversal through the unfinished linker SMILES.
    parts = []
    if 0 in mods_a:
        parts.append(mod_dict[mods_a[0]]['SMILES-Front'])
    parts.extend(fragment(aa, pos, mods_a) for pos, aa in enumerate(alpha[:pos_a - 1], 1))
    parts.append(linkers[key])
    parts.extend(fragment(beta[pos - 1], pos, mods_b, True) for pos in range(pos_b - 1, 0, -1))
    if 0 in mods_b:
        parts.append(mod_dict[mods_b[0]]['SMILES-Reverse'])
    parts.append(')C(=O)')
    parts.extend(fragment(aa, pos, mods_b) for pos, aa in enumerate(beta[pos_b:], pos_b + 1))
    parts.append('O)C(=O)')
    parts.extend(fragment(aa, pos, mods_a) for pos, aa in enumerate(alpha[pos_a:], pos_a + 1))
    parts.append('O')
    fragments = residue_fragments(alpha, mods_a) + [None] + residue_fragments(beta, mods_b)
    return ''.join(parts), fragments


def composition(fragments, pad_to=None):
    rows = []
    for fragment in fragments:
        counts = [0] * 118
        if fragment is not None:
            molecule = Chem.MolFromSmiles(fragment)
            if molecule is None:
                raise ValueError("Invalid residue SMILES: " + fragment)
            for atom in Chem.AddHs(molecule).GetAtoms():
                counts[ELEMENT_INDEX[atom.GetSymbol()]] += 1
        rows.append(counts)
    if pad_to is not None:
        if len(rows) > pad_to:
            raise ValueError("Refusing to silently truncate composition features")
        rows += [[0] * 118 for _ in range(pad_to - len(rows))]
    return torch.tensor(rows, dtype=torch.float32)


@lru_cache(maxsize=int(os.environ.get('PPRED_FEATURE_CACHE_SIZE', '256')))
def molecular_features(task, sequence, modifications='', linker=''):
    if task == 'crosslink_rt':
        smiles, fragments = crosslink_structure(sequence, modifications, linker)
    else:
        smiles, fragments = single_structure(sequence, modifications)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Could not construct a molecular graph for " + sequence)
    nodes = [atom_to_feature_vector(atom) for atom in molecule.GetAtoms()]
    edges, attributes = [], []
    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_to_feature_vector(bond)
        edges.extend([(start, end), (end, start)])
        attributes.extend([feature, feature])
    fingerprint = (list(AllChem.GetMACCSKeysFingerprint(molecule))
                   + list(AllChem.GetErGFingerprint(molecule, fuzzIncrement=0.3, maxPath=21, minPath=1))
                   + list(GetPubChemFPs(molecule)))
    if len(fingerprint) != 1489:
        raise ValueError("Fingerprint layout is incompatible with the source checkpoint")
    return Data(
        x=torch.tensor(nodes, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous().reshape(2, -1),
        edge_attr=torch.tensor(attributes, dtype=torch.long).reshape(-1, 3),
        fp=torch.tensor(fingerprint, dtype=torch.float32),
        element_composition_array=composition(fragments, None if task == 'ms2' else 60),
        smiles=smiles,
    )
