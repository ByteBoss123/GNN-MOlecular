"""Load ESOL and Tox21, featurize molecules for GNNs and fingerprint baselines, and make scaffold splits."""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

TOX21_TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
               "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Na", "other"]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
       Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]


def one_hot(x, choices):
    v = [0] * (len(choices) + 1)
    v[choices.index(x) if x in choices else len(choices)] = 1
    return v


def atom_features(a):
    sym = a.GetSymbol() if a.GetSymbol() in ATOMS[:-1] else "other"
    return (one_hot(sym, ATOMS[:-1]) + one_hot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
            + one_hot(a.GetFormalCharge(), [-1, 0, 1]) + one_hot(a.GetTotalNumHs(), [0, 1, 2, 3])
            + one_hot(a.GetHybridization(), HYB) + [int(a.GetIsAromatic()), int(a.IsInRing())])


N_ATOM_FEATURES = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def mol_to_graph(mol, y):
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float))


def fingerprint(mol, n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load(name):
    """Returns (mols, y, smiles). y is (n,) for ESOL and (n, 12) with NaN for missing Tox21 labels."""
    if name == "esol":
        df = pd.read_csv(os.path.join(RAW, "delaney-processed.csv"))
        smiles, y = df["smiles"].str.strip(), df["measured log solubility in mols per litre"].values
    elif name == "tox21":
        df = pd.read_csv(os.path.join(RAW, "tox21.csv.gz"))
        smiles, y = df["smiles"], df[TOX21_TASKS].values.astype(float)
    else:
        raise ValueError(name)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None and m.GetNumAtoms() > 0]
    return [mols[i] for i in keep], y[keep], [smiles.iloc[i] for i in keep]


def scaffold_split(smiles, frac=(0.8, 0.1, 0.1)):
    """Deterministic Bemis-Murcko scaffold split: whole scaffold groups go to one set, largest first.
    Test molecules have ring systems never seen in training, which is harder and more realistic than random."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(smiles)
    train, val, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= frac[0] * n:
            train += g
        elif len(val) + len(g) <= frac[1] * n:
            val += g
        else:
            test += g
    return np.array(train), np.array(val), np.array(test)
