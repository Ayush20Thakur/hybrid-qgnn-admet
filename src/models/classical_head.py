"""Classical GNN baseline models: Full MLP head and Parameter-Matched head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.gnn_encoder import MolecularGNNEncoder


def quantum_readout_param_count(
    n_qubits: int = 4,
    n_layers: int = 3,
    num_tasks: int = 1,
) -> int:
    """Exact trainable parameter count of a StronglyEntanglingLayers + Linear readout."""
    circuit_params = 3 * n_qubits * n_layers
    linear_params = n_qubits * num_tasks + num_tasks
    return circuit_params + linear_params


class ClassicalGNNFull(nn.Module):
    """Classical GNN baseline with a standard 2-layer MLP classifier head."""

    def __init__(
        self,
        encoder: MolecularGNNEncoder,
        mlp_hidden: int = 32,
        num_tasks: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.latent_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_tasks),
        )

    def forward(self, data) -> torch.Tensor:
        latent = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(latent)


class ClassicalGNNParamMatched(nn.Module):
    """Classical GNN baseline with a readout head matching the quantum parameter budget."""

    def __init__(
        self,
        encoder: MolecularGNNEncoder,
        n_layers: int = 3,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        n_qubits = encoder.latent_dim
        target_params = quantum_readout_param_count(
            n_qubits=n_qubits, n_layers=n_layers, num_tasks=num_tasks
        )

        # Solve for hidden width h so Linear(n_qubits, h, bias=...) + Linear(h, num_tasks)
        # matches target_params (for 4 qubits, 3 layers, 1 task: 4*8 + 8*1 + 1 = 41 exact).
        best_cfg = None
        best_diff = 10**9
        for use_bias1 in (False, True):
            per_h = n_qubits + (1 if use_bias1 else 0) + num_tasks
            h = max(1, round((target_params - num_tasks) / per_h))
            p_count = h * per_h + num_tasks
            diff = abs(p_count - target_params)
            if diff < best_diff:
                best_diff = diff
                best_cfg = (h, use_bias1)

        h, use_bias1 = best_cfg
        self.head = nn.Sequential(
            nn.Linear(n_qubits, h, bias=use_bias1),
            nn.ReLU(),
            nn.Linear(h, num_tasks),
        )

    def forward(self, data) -> torch.Tensor:
        latent = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(latent)


def count_parameters(model: nn.Module, verbose: bool = True) -> dict[str, int]:
    """Return and optionally print encoder, head, and total trainable parameter counts."""
    enc_params = (
        sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        if hasattr(model, "encoder")
        else 0
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = total_params - enc_params

    if verbose:
        print(
            f"[{model.__class__.__name__}] "
            f"encoder={enc_params:,} | head={head_params:,} | total={total_params:,}"
        )
    return {"encoder": enc_params, "head": head_params, "total": total_params}
