"""RDKit molecular graph featurizer for MoleculeNet (BBBP, BACE) datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondStereo, BondType, ChiralType, HybridizationType
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet

logger = logging.getLogger(__name__)

# Vocabulary definitions for one-hot encodings
ATOM_SYMBOLS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", "B", "Si", "Se"]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]
TOTAL_HS = [0, 1, 2, 3, 4]
CHIRAL_TAGS = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
]

BOND_TYPES = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]
BOND_STEREOS = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
]

DATASET_CSV_SPEC = {
    "bbbp": {"pyg_name": "BBBP", "csv_name": "BBBP.csv", "smiles_col": "smiles", "label_col": "p_np"},
    "bace": {"pyg_name": "BACE", "csv_name": "bace.csv", "smiles_col": "mol", "label_col": "Class"},
}


def _one_hot_with_unknown(value: object, choices: list[object]) -> list[float]:
    """Return a one-hot vector with an extra trailing slot for unknown values."""
    vec = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    vec[idx] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    """Compute numeric feature vector for a single RDKit atom."""
    feats: list[float] = []
    feats.extend(_one_hot_with_unknown(atom.GetSymbol(), ATOM_SYMBOLS))
    feats.extend(_one_hot_with_unknown(atom.GetTotalDegree(), ATOM_DEGREES))
    feats.extend(_one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGES))
    feats.extend(_one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS))
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.extend(_one_hot_with_unknown(atom.GetTotalNumHs(), TOTAL_HS))
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats.extend(_one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS))
    return feats


def bond_features(bond: Chem.Bond) -> list[float]:
    """Compute numeric feature vector for a single RDKit bond."""
    feats: list[float] = []
    feats.extend(_one_hot_with_unknown(bond.GetBondType(), BOND_TYPES))
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    feats.extend(_one_hot_with_unknown(bond.GetStereo(), BOND_STEREOS))
    return feats


def get_feature_dims() -> tuple[int, int]:
    """Return exact (node_dim, edge_dim) produced by this featurizer."""
    node_dim = (
        (len(ATOM_SYMBOLS) + 1)
        + (len(ATOM_DEGREES) + 1)
        + (len(FORMAL_CHARGES) + 1)
        + (len(HYBRIDIZATIONS) + 1)
        + 1
        + (len(TOTAL_HS) + 1)
        + 1
        + (len(CHIRAL_TAGS) + 1)
    )
    edge_dim = (len(BOND_TYPES) + 1) + 1 + 1 + (len(BOND_STEREOS) + 1)
    return node_dim, edge_dim


def smiles_to_graph(smiles: str, label: float | None = None) -> Data | None:
    """Convert a SMILES string into a PyG Data object, or None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(b_feat)
        edge_attrs.append(b_feat)

    _, edge_dim = get_feature_dims()
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if label is not None:
        data.y = torch.tensor([[float(label)]], dtype=torch.float32)
    return data


def featurize_smiles_list(
    smiles_list: Iterable[str],
    labels: Iterable[float],
    quiet_rdkit: bool = True,
) -> tuple[list[Data], int]:
    """Featurize paired SMILES and labels, skipping invalid molecules."""
    if quiet_rdkit:
        RDLogger.DisableLog("rdApp.*")

    graphs: list[Data] = []
    skipped = 0
    for idx, (smi, lbl) in enumerate(zip(smiles_list, labels)):
        if pd.isna(lbl):
            skipped += 1
            logger.warning("Skipping row %d with NaN label (SMILES=%r)", idx, smi)
            continue
        graph = smiles_to_graph(str(smi), float(lbl))
        if graph is None:
            skipped += 1
            logger.warning("Skipping invalid SMILES at row %d: %r", idx, smi)
            continue
        graphs.append(graph)

    if quiet_rdkit:
        RDLogger.EnableLog("rdApp.*")
    return graphs, skipped


def _ensure_raw_csv(dataset_key: str, raw_dir: Path) -> Path:
    """Ensure the MoleculeNet raw CSV exists via PyG MoleculeNet download."""
    spec = DATASET_CSV_SPEC[dataset_key]
    raw_dir.mkdir(parents=True, exist_ok=True)
    # PyG MoleculeNet stores raw CSV at <raw_dir>/<key>/raw/<csv_name>
    pyg_ds = MoleculeNet(root=str(raw_dir), name=spec["pyg_name"])
    csv_path = Path(pyg_ds.raw_dir) / spec["csv_name"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected MoleculeNet CSV at {csv_path}")
    return csv_path


def load_moleculenet_dataset(
    name: str,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    force_reload: bool = False,
) -> list[Data]:
    """Load, featurize, and cache MoleculeNet BBBP or BACE as PyG Data objects."""
    key = name.strip().lower()
    if key not in DATASET_CSV_SPEC:
        raise ValueError(f"Unsupported dataset {name!r}. Choose from {list(DATASET_CSV_SPEC)}.")

    processed_path = Path(processed_dir) / f"{key}_graphs.pt"
    if processed_path.exists() and not force_reload:
        return torch.load(processed_path, weights_only=False)

    spec = DATASET_CSV_SPEC[key]
    csv_path = _ensure_raw_csv(key, Path(raw_dir))
    df = pd.read_csv(csv_path)

    graphs, skipped = featurize_smiles_list(
        df[spec["smiles_col"]].tolist(),
        df[spec["label_col"]].tolist(),
    )
    if skipped > 0:
        logger.info("[%s] Featurized %d molecules (%d invalid/skipped).", key.upper(), len(graphs), skipped)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, processed_path)
    return graphs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    n_dim, e_dim = get_feature_dims()
    print(f"Feature dims -> node_dim={n_dim}, edge_dim={e_dim}")
    for ds_name in ("bbbp", "bace"):
        ds = load_moleculenet_dataset(ds_name, force_reload=True)
        pos_rate = sum(int(g.y.item() == 1.0) for g in ds) / len(ds)
        print(f"{ds_name.upper()}: {len(ds)} graphs | positive_rate={pos_rate:.4f}")
