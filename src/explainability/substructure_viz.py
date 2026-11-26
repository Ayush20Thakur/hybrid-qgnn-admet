"""Atom-level gradient saliency explainability and RDKit 2D substructure rendering."""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.data.featurizer import get_feature_dims, load_moleculenet_dataset, smiles_to_graph
from src.data.scaffold_split import load_split_indices
from src.models.classical_head import ClassicalGNNParamMatched
from src.models.gnn_encoder import MolecularGNNEncoder
from src.models.hybrid_qgnn import HybridQGNN

warnings.filterwarnings("ignore", message="The usage of `scatter\\(reduce='max'\\)`")


def find_best_checkpoint(
    dataset: str,
    model_name: str = "hybrid_angle",
    metrics_dir: str | Path = "results/metrics",
    checkpoints_dir: str | Path = "results/checkpoints",
) -> Path | None:
    """Find the checkpoint with highest validation ROC-AUC for (dataset, model_name)."""
    ds = dataset.lower()
    best_auc = -1.0
    best_seed = None
    for p in sorted(Path(metrics_dir).glob(f"{ds}_{model_name}_scaffold_f100_seed*.json")):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        val_auc = float(data["val"]["roc_auc"])
        if val_auc > best_auc:
            best_auc = val_auc
            best_seed = int(data["seed"])

    if best_seed is None:
        return None
    ckpt = Path(checkpoints_dir) / f"{ds}_{model_name}_seed{best_seed}.pt"
    return ckpt if ckpt.exists() else None


def load_trained_model(
    dataset: str,
    model_name: str = "hybrid_angle",
) -> tuple[torch.nn.Module | None, Path | None]:
    """Load best checkpoint for (dataset, model_name), returning (None, None) if missing."""
    ckpt_path = find_best_checkpoint(dataset=dataset, model_name=model_name)
    if ckpt_path is None:
        return None, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    node_dim, edge_dim = get_feature_dims()
    encoder = MolecularGNNEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=int(ckpt.get("hidden_dim", 64)),
        latent_dim=int(ckpt.get("n_qubits", 4)),
    )
    if model_name in ("hybrid_angle", "hybrid_reuploading"):
        enc_type = "angle" if model_name == "hybrid_angle" else "reuploading"
        model: torch.nn.Module = HybridQGNN(
            encoder=encoder,
            n_layers=int(ckpt.get("n_layers", 3)),
            num_tasks=1,
            encoding_type=enc_type,
            quantum_device="default.qubit",
        )
    elif model_name == "classical_matched":
        model = ClassicalGNNParamMatched(
            encoder=encoder,
            n_layers=int(ckpt.get("n_layers", 3)),
            num_tasks=1,
        )
    else:
        raise ValueError(f"Unsupported model_name {model_name!r}")

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt_path


def compute_physicochemical_props(mol: Chem.Mol) -> dict[str, float | int | str]:
    """Compute Bemis-Murcko scaffold and standard ADMET physicochemical descriptors."""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return {
        "scaffold": scaffold if scaffold else "(Acyclic / No Ring Scaffold)",
        "MW": round(float(Descriptors.MolWt(mol)), 2),
        "LogP": round(float(Crippen.MolLogP(mol)), 2),
        "TPSA": round(float(rdMolDescriptors.CalcTPSA(mol)), 2),
        "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": int(rdMolDescriptors.CalcNumHBA(mol)),
    }


def _render_saliency_drawing(
    mol: Chem.Mol,
    atom_scores: np.ndarray,
    as_svg: bool = True,
    width: int = 420,
    height: int = 300,
    top_fraction: float = 0.4,
) -> str | bytes:
    """Render RDKit 2D structure highlighting top contributing atoms in teal/blue."""
    n_atoms = mol.GetNumAtoms()
    k = max(1, int(np.ceil(top_fraction * n_atoms)))
    top_atoms = set(np.argsort(atom_scores)[-k:].tolist())

    atom_colors: dict[int, tuple[float, float, float]] = {}
    for idx in top_atoms:
        intensity = float(atom_scores[idx])
        # Teal-blue highlight scaled by normalized saliency
        atom_colors[int(idx)] = (
            max(0.15, 0.85 - 0.70 * intensity),
            max(0.55, 0.95 - 0.30 * intensity),
            max(0.65, 0.98 - 0.20 * intensity),
        )

    bond_list: list[int] = []
    bond_colors: dict[int, tuple[float, float, float]] = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in top_atoms and a2 in top_atoms:
            b_idx = bond.GetIdx()
            bond_list.append(b_idx)
            bond_colors[b_idx] = (0.20, 0.68, 0.72)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height) if as_svg else rdMolDraw2D.MolDraw2DCairo(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=sorted(top_atoms),
        highlightAtomColors=atom_colors,
        highlightBonds=bond_list,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def explain_molecule(
    smiles: str,
    model: HybridQGNN,
    width: int = 440,
    height: int = 310,
) -> dict[str, Any]:
    """Compute prediction, quantum internals, atom saliency, properties, and 2D SVG."""
    mol = Chem.MolFromSmiles(smiles)
    graph = smiles_to_graph(smiles)
    if mol is None or graph is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    model.eval()
    x = graph.x.clone().detach().requires_grad_(True)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)

    latent = model.encoder(x, graph.edge_index, graph.edge_attr, batch)
    q_exp = model.quantum_layer(latent)
    logit = model.classifier(q_exp)
    prob = float(torch.sigmoid(logit).item())

    model.zero_grad()
    logit.sum().backward()
    raw_grad = x.grad.norm(p=2, dim=-1).detach().cpu().numpy()
    max_g = float(raw_grad.max())
    norm_saliency = raw_grad / max_g if max_g > 0 else np.zeros_like(raw_grad)

    svg = _render_saliency_drawing(mol, norm_saliency, as_svg=True, width=width, height=height)
    png_bytes = _render_saliency_drawing(mol, norm_saliency, as_svg=False, width=width, height=height)
    props = compute_physicochemical_props(mol)

    return {
        "smiles": smiles,
        "probability": prob,
        "logit": float(logit.item()),
        "latent_angles": latent.detach().cpu().numpy().ravel().tolist(),
        "pauli_z_expvals": q_exp.detach().cpu().numpy().ravel().tolist(),
        "atom_saliency": norm_saliency.tolist(),
        "svg": svg,
        "png_bytes": png_bytes,
        "properties": props,
    }


def export_explainability_grid(
    out_path: str | Path = "results/figures/explainability_substructures.png",
) -> Path:
    """Export a 300-DPI 4-molecule explainability grid (2 BACE actives + 2 BBBP permeables)."""
    bace_model, _ = load_trained_model("bace", "hybrid_angle")
    bbbp_model, _ = load_trained_model("bbbp", "hybrid_angle")
    if bace_model is None or bbbp_model is None:
        raise RuntimeError("Trained BACE and BBBP hybrid_angle checkpoints are required.")

    samples: list[tuple[str, str, HybridQGNN, str]] = []
    for ds_name, mdl, task_label in (
        ("bace", bace_model, "BACE-1 Active"),
        ("bbbp", bbbp_model, "BBBP Permeable"),
    ):
        graphs = load_moleculenet_dataset(ds_name)
        _, _, test_idx = load_split_indices(ds_name, split_type="scaffold")
        count = 0
        for idx in test_idx:
            g = graphs[int(idx)]
            if int(g.y.item()) == 1 and g.num_nodes <= 35:
                samples.append((ds_name.upper(), task_label, mdl, g.smiles))
                count += 1
                if count == 2:
                    break

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for ax, (ds_label, task_label, mdl, smi) in zip(axes.ravel(), samples):
        res = explain_molecule(smi, mdl, width=520, height=340)
        img = Image.open(io.BytesIO(res["png_bytes"]))
        ax.imshow(img)
        ax.axis("off")
        scaff = res["properties"]["scaffold"]
        if len(scaff) > 36:
            scaff = scaff[:33] + "..."
        ax.set_title(
            f"{ds_label} ({task_label}) | P(Active) = {res['probability']:.3f}\n"
            f"Scaffold: {scaff}",
            fontsize=10,
            pad=6,
        )

    fig.suptitle(
        "Hybrid QGNN Atom-Level Gradient Saliency on Unseen Test Scaffolds",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved explainability grid: {out_file}")
    return out_file


if __name__ == "__main__":
    export_explainability_grid()
