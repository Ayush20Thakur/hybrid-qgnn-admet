"""Deterministic Bemis-Murcko scaffold splitting and random-split baseline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Extract canonical Bemis-Murcko scaffold SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES encountered during scaffold extraction: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def bemis_murcko_scaffold_split(
    data_list: Sequence[Data],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    include_chirality: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split molecules 80/10/10 by Bemis-Murcko core scaffold.

    Scaffold groups are sorted largest-first (tie-broken by scaffold SMILES) and
    assigned whole-group to train, validation, and test splits so that no
    scaffold appears in more than one split.
    """
    if not np.isclose(frac_train + frac_val + frac_test, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")

    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, data in enumerate(data_list):
        scaff = generate_scaffold(data.smiles, include_chirality=include_chirality)
        scaffold_to_indices[scaff].append(idx)

    # Largest scaffold groups first; tie-break deterministically by scaffold SMILES
    sorted_groups = sorted(
        scaffold_to_indices.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )

    n_total = len(data_list)
    train_cutoff = frac_train * n_total
    val_cutoff = (frac_train + frac_val) * n_total

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _, indices in sorted_groups:
        if len(train_idx) + len(indices) <= train_cutoff or len(train_idx) == 0:
            train_idx.extend(indices)
        elif len(train_idx) + len(val_idx) + len(indices) <= val_cutoff or len(val_idx) == 0:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def random_split(
    n_samples: int,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded random 80/10/10 split for Random-vs-Scaffold leakage ablation."""
    if not np.isclose(frac_train + frac_val + frac_test, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(round(frac_train * n_samples))
    n_val = int(round(frac_val * n_samples))

    train_idx = perm[:n_train].astype(np.int64)
    val_idx = perm[n_train : n_train + n_val].astype(np.int64)
    test_idx = perm[n_train + n_val :].astype(np.int64)
    return train_idx, val_idx, test_idx


def verify_zero_scaffold_overlap(
    data_list: Sequence[Data],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    include_chirality: bool = False,
) -> dict[str, set[str]]:
    """Assert zero molecule-index and zero Bemis-Murcko scaffold overlap across splits."""
    idx_sets = [set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())]
    assert len(idx_sets[0] & idx_sets[1]) == 0, "Index overlap between train and val!"
    assert len(idx_sets[0] & idx_sets[2]) == 0, "Index overlap between train and test!"
    assert len(idx_sets[1] & idx_sets[2]) == 0, "Index overlap between val and test!"
    assert sum(len(s) for s in idx_sets) == len(data_list), "Split indices do not cover all molecules!"

    def _scaffolds(indices: np.ndarray) -> set[str]:
        return {
            generate_scaffold(data_list[int(i)].smiles, include_chirality=include_chirality)
            for i in indices
        }

    train_scaffs = _scaffolds(train_idx)
    val_scaffs = _scaffolds(val_idx)
    test_scaffs = _scaffolds(test_idx)

    assert len(train_scaffs & val_scaffs) == 0, "Scaffold leakage between train and val!"
    assert len(train_scaffs & test_scaffs) == 0, "Scaffold leakage between train and test!"
    assert len(val_scaffs & test_scaffs) == 0, "Scaffold leakage between val and test!"

    return {"train": train_scaffs, "val": val_scaffs, "test": test_scaffs}


def save_split_indices(
    dataset_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_type: str = "scaffold",
    processed_dir: str | Path = "data/processed",
) -> Path:
    """Save train/val/test index arrays as .npy files under data/processed/."""
    out_dir = Path(processed_dir) / f"{dataset_name.lower()}_{split_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)
    np.save(out_dir / "test_idx.npy", test_idx)
    return out_dir


def load_split_indices(
    dataset_name: str,
    split_type: str = "scaffold",
    processed_dir: str | Path = "data/processed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved train/val/test index arrays from data/processed/."""
    in_dir = Path(processed_dir) / f"{dataset_name.lower()}_{split_type}"
    return (
        np.load(in_dir / "train_idx.npy"),
        np.load(in_dir / "val_idx.npy"),
        np.load(in_dir / "test_idx.npy"),
    )


def summarize_splits(
    dataset_name: str,
    data_list: Sequence[Data],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    scaffolds: dict[str, set[str]],
) -> None:
    """Print molecule counts, scaffold counts, and positive-class ratios per split."""
    print(f"\n=== {dataset_name.upper()} Bemis-Murcko Scaffold Split Summary ===")
    print(f"{'Split':<8} | {'Molecules':>9} | {'Pct':>6} | {'Scaffolds':>9} | {'Positives':>9} | {'Pos Ratio':>9}")
    print("-" * 64)
    n_total = len(data_list)
    for name, idx in (("Train", train_idx), ("Val", val_idx), ("Test", test_idx)):
        pos = sum(int(data_list[int(i)].y.item() == 1.0) for i in idx)
        ratio = pos / len(idx) if len(idx) > 0 else 0.0
        pct = 100.0 * len(idx) / n_total
        print(
            f"{name:<8} | {len(idx):>9d} | {pct:>5.1f}% | {len(scaffolds[name.lower()]):>9d} | {pos:>9d} | {ratio:>9.4f}"
        )


if __name__ == "__main__":
    import logging
    from src.data.featurizer import get_feature_dims, load_moleculenet_dataset

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    node_dim, edge_dim = get_feature_dims()
    print(f"RDKit Graph Featurizer dims: node_dim={node_dim}, edge_dim={edge_dim}")

    for ds_name in ("bbbp", "bace"):
        graphs = load_moleculenet_dataset(ds_name, force_reload=True)
        tr, va, te = bemis_murcko_scaffold_split(graphs)
        scaffs = verify_zero_scaffold_overlap(graphs, tr, va, te)
        out_dir = save_split_indices(ds_name, tr, va, te, split_type="scaffold")
        summarize_splits(ds_name, graphs, tr, va, te, scaffs)
        print(f"Saved split indices to: {out_dir}")
        print(f"Zero scaffold overlap assertion PASSED for {ds_name.upper()}.")
