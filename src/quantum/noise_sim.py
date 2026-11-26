"""NISQ density-matrix noise sweep and Zero-Noise Extrapolation (ZNE) evaluation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.data.featurizer import get_feature_dims, load_moleculenet_dataset
from src.data.scaffold_split import load_split_indices
from src.models.gnn_encoder import MolecularGNNEncoder
from src.models.hybrid_qgnn import HybridQGNN

warnings.filterwarnings("ignore", message="The usage of `scatter\\(reduce='max'\\)`")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def find_best_bace_checkpoint(
    metrics_dir: str | Path = "results/metrics",
    checkpoints_dir: str | Path = "results/checkpoints",
) -> Path:
    """Select the BACE hybrid_angle checkpoint with highest validation ROC-AUC."""
    best_auc = -1.0
    best_seed = None
    for p in sorted(Path(metrics_dir).glob("bace_hybrid_angle_scaffold_f100_seed*.json")):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        val_auc = float(data["val"]["roc_auc"])
        if val_auc > best_auc:
            best_auc = val_auc
            best_seed = int(data["seed"])
    if best_seed is None:
        raise FileNotFoundError("No BACE hybrid_angle metric JSONs found.")
    ckpt = Path(checkpoints_dir) / f"bace_hybrid_angle_seed{best_seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint at {ckpt}")
    print(f"Selected best BACE checkpoint: {ckpt} (val_roc_auc={best_auc:.4f})")
    return ckpt


def load_frozen_bace_model(ckpt_path: Path) -> tuple[HybridQGNN, np.ndarray, np.ndarray, np.ndarray]:
    """Load HybridQGNN checkpoint and extract BACE test embeddings, labels, and weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    node_dim, edge_dim = get_feature_dims()
    encoder = MolecularGNNEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=int(ckpt.get("hidden_dim", 64)),
        latent_dim=int(ckpt.get("n_qubits", 4)),
    )
    model = HybridQGNN(
        encoder=encoder,
        n_layers=int(ckpt.get("n_layers", 3)),
        num_tasks=1,
        encoding_type="angle",
        quantum_device="default.qubit",
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    graphs = load_moleculenet_dataset("bace")
    _, _, test_idx = load_split_indices("bace", split_type="scaffold")
    test_graphs = [graphs[int(i)] for i in test_idx]
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    latents: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            z = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            latents.append(z.cpu().numpy())
            labels.append(batch.y.view(-1).cpu().numpy())

    x_latent = np.concatenate(latents, axis=0)
    y_true = np.concatenate(labels, axis=0)
    q_weights = model.quantum_layer.q_layer.weights.detach().cpu().numpy()
    return model, x_latent, y_true, q_weights


def _apply_unitary(inputs: np.ndarray, weights: np.ndarray, p_depol: float, n_qubits: int) -> None:
    """Apply AngleEmbedding(Y) + StronglyEntanglingLayers with depolarizing noise."""
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
        if p_depol > 0.0:
            qml.DepolarizingChannel(p_depol, wires=i)

    n_layers = weights.shape[0]
    for l_idx in range(n_layers):
        r = (l_idx % (n_qubits - 1)) + 1
        for i in range(n_qubits):
            phi, theta, omega = weights[l_idx, i]
            qml.Rot(phi, theta, omega, wires=i)
            if p_depol > 0.0:
                qml.DepolarizingChannel(p_depol, wires=i)
        for i in range(n_qubits):
            target = (i + r) % n_qubits
            qml.CNOT(wires=[i, target])
            if p_depol > 0.0:
                qml.DepolarizingChannel(p_depol, wires=i)
                qml.DepolarizingChannel(p_depol, wires=target)


def _apply_unitary_adjoint(inputs: np.ndarray, weights: np.ndarray, p_depol: float, n_qubits: int) -> None:
    """Apply adjoint U^dagger with depolarizing noise for unitary folding."""
    n_layers = weights.shape[0]
    for l_idx in range(n_layers - 1, -1, -1):
        r = (l_idx % (n_qubits - 1)) + 1
        for i in range(n_qubits - 1, -1, -1):
            target = (i + r) % n_qubits
            qml.CNOT(wires=[i, target])
            if p_depol > 0.0:
                qml.DepolarizingChannel(p_depol, wires=i)
                qml.DepolarizingChannel(p_depol, wires=target)
        for i in range(n_qubits - 1, -1, -1):
            phi, theta, omega = weights[l_idx, i]
            qml.Rot(-omega, -theta, -phi, wires=i)
            if p_depol > 0.0:
                qml.DepolarizingChannel(p_depol, wires=i)

    for i in range(n_qubits - 1, -1, -1):
        qml.RY(-inputs[i], wires=i)
        if p_depol > 0.0:
            qml.DepolarizingChannel(p_depol, wires=i)


def make_noisy_qnode(n_qubits: int = 4):
    """Build a default.mixed QNode supporting depolarizing noise, readout BitFlip, and folding."""
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def noisy_circuit(inputs, weights, p_depol: float, p_readout: float, scale: int):
        _apply_unitary(inputs, weights, p_depol, n_qubits)
        if scale == 3:
            _apply_unitary_adjoint(inputs, weights, p_depol, n_qubits)
            _apply_unitary(inputs, weights, p_depol, n_qubits)
        elif scale != 1:
            raise ValueError(f"Unsupported folding scale {scale}")

        if p_readout > 0.0:
            for i in range(n_qubits):
                qml.BitFlip(p_readout, wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return noisy_circuit


def evaluate_expectations(
    model: HybridQGNN,
    exps: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float]:
    """Pass (N, n_qubits) Pauli-Z expectations through frozen classifier and score ROC/PR-AUC."""
    with torch.no_grad():
        logits = model.classifier(torch.tensor(exps, dtype=torch.float32))
        probs = torch.sigmoid(logits).view(-1).numpy()
    return float(roc_auc_score(y_true, probs)), float(average_precision_score(y_true, probs))


def main() -> None:
    ckpt_path = find_best_bace_checkpoint()
    model, x_latent, y_true, q_weights = load_frozen_bace_model(ckpt_path)
    n_test, n_qubits = x_latent.shape
    print(f"Loaded {n_test} BACE scaffold test molecules (positives={int(y_true.sum())}).")

    qnode = make_noisy_qnode(n_qubits=n_qubits)

    # Verify folding equivalence at p=0.0
    e_s1 = np.array(qnode(x_latent[0], q_weights, 0.0, 0.0, 1), dtype=float)
    e_s3 = np.array(qnode(x_latent[0], q_weights, 0.0, 0.0, 3), dtype=float)
    assert np.max(np.abs(e_s1 - e_s3)) < 1e-6, "Unitary folding mismatch at p=0!"

    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    rows: list[dict[str, Any]] = []
    ideal_exps: np.ndarray | None = None

    for p in noise_levels:
        p_ro = 0.01 if p > 0.0 else 0.0
        exps_s1 = np.array([qnode(x, q_weights, p, p_ro, 1) for x in x_latent], dtype=float)
        exps_s3 = np.array([qnode(x, q_weights, p, p_ro, 3) for x in x_latent], dtype=float)
        # Linear Richardson extrapolation from scale_factors=[1, 3] to 0
        exps_zne = np.clip(1.5 * exps_s1 - 0.5 * exps_s3, -1.0, 1.0)

        if ideal_exps is None:
            ideal_exps = exps_s1

        roc_unmit, pr_unmit = evaluate_expectations(model, exps_s1, y_true)
        roc_zne, pr_zne = evaluate_expectations(model, exps_zne, y_true)
        mae_unmit = float(np.mean(np.abs(exps_s1 - ideal_exps)))
        mae_zne = float(np.mean(np.abs(exps_zne - ideal_exps)))

        rows.append({
            "p_depol": p,
            "p_readout": p_ro,
            "roc_auc_unmitigated": roc_unmit,
            "pr_auc_unmitigated": pr_unmit,
            "mae_unmitigated": mae_unmit,
            "roc_auc_zne": roc_zne,
            "pr_auc_zne": pr_zne,
            "mae_zne": mae_zne,
        })

    df = pd.DataFrame(rows)
    out_csv = Path("results/metrics/nisq_noise_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # 300-DPI plot
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(df["p_depol"], df["roc_auc_unmitigated"], marker="o", lw=2, color="#d95f02", label="Unmitigated (NISQ)")
    axes[0].plot(df["p_depol"], df["roc_auc_zne"], marker="s", lw=2, color="#1b9e77", label="ZNE (Scale [1, 3])")
    axes[0].set_title("BACE Test ROC-AUC Under NISQ Noise")
    axes[0].set_xlabel("Depolarizing Error Rate (p)")
    axes[0].set_ylabel("Test ROC-AUC")
    axes[0].legend(frameon=True)

    axes[1].plot(df["p_depol"], df["mae_unmitigated"], marker="o", lw=2, color="#d95f02", label="Unmitigated MAE")
    axes[1].plot(df["p_depol"], df["mae_zne"], marker="s", lw=2, color="#1b9e77", label="ZNE MAE")
    axes[1].set_title("Pauli-Z Expectation Error vs. Ideal")
    axes[1].set_xlabel("Depolarizing Error Rate (p)")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    out_fig = Path("results/figures/nisq_noise_zne.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n=== NISQ NOISE SWEEP & ZNE MITIGATION (BACE Scaffold Test, N=152) ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv} and {out_fig}")


if __name__ == "__main__":
    main()
