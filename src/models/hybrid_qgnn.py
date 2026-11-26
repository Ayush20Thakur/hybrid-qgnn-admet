"""End-to-end Hybrid Quantum-Classical Graph Neural Network (HybridQGNN)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.models.gnn_encoder import MolecularGNNEncoder
from src.models.quantum_layer import QuantumReadoutLayer


class HybridQGNN(nn.Module):
    """MolecularGNNEncoder -> QuantumReadoutLayer -> Linear readout."""

    def __init__(
        self,
        encoder: MolecularGNNEncoder,
        n_layers: int = 3,
        num_tasks: int = 1,
        encoding_type: Literal["angle", "reuploading"] = "angle",
        quantum_device: str = "lightning.qubit",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        n_qubits = encoder.latent_dim
        self.quantum_layer = QuantumReadoutLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_type=encoding_type,
            device_name=quantum_device,
        )
        self.classifier = nn.Linear(n_qubits, num_tasks)

    def to(self, *args, **kwargs):
        """Move classical modules to target device while keeping quantum TorchLayer on CPU."""
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.encoder.to(*args, **kwargs)
        self.classifier.to(*args, **kwargs)
        # Keep quantum_layer on CPU so lightning.qubit and optimizer param device stay consistent
        if dtype is not None:
            self.quantum_layer.to(dtype=dtype)
        return self

    def forward(self, data) -> torch.Tensor:
        latent = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        q_out = self.quantum_layer(latent)
        return self.classifier(q_out)


if __name__ == "__main__":
    from torch_geometric.data import Batch, Data
    from src.models.classical_head import count_parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running HybridQGNN unit test on device={device}")

    # Create 2 synthetic molecular graphs (45-dim node features, 12-dim edge features)
    g1 = Data(
        x=torch.randn(5, 45),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.randn(4, 12),
        y=torch.tensor([[1.0]]),
    )
    g2 = Data(
        x=torch.randn(4, 45),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 12),
        y=torch.tensor([[0.0]]),
    )
    batch = Batch.from_data_list([g1, g2]).to(device)

    for enc_type in ("angle", "reuploading"):
        encoder = MolecularGNNEncoder(node_dim=45, edge_dim=12, hidden_dim=64, latent_dim=4)
        model = HybridQGNN(encoder=encoder, n_layers=3, num_tasks=1, encoding_type=enc_type).to(device)
        counts = count_parameters(model)
        assert counts["head"] == 41, f"Expected 41 head params, got {counts['head']}"

        logits = model(batch)
        assert logits.shape == (2, 1), f"Unexpected logit shape: {logits.shape}"
        loss = nn.BCEWithLogitsLoss()(logits, batch.y)
        loss.backward()

        enc_grad = model.encoder.node_proj.weight.grad
        q_grad = next(model.quantum_layer.parameters()).grad
        cls_grad = model.classifier.weight.grad
        assert enc_grad is not None and enc_grad.norm().item() > 0, "Missing encoder gradients"
        assert q_grad is not None and torch.isfinite(q_grad).all(), "Missing quantum gradients"
        assert cls_grad is not None and cls_grad.norm().item() > 0, "Missing classifier gradients"
        print(
            f"  [{enc_type}] PASS | logits={tuple(logits.shape)} | "
            f"enc_grad_norm={enc_grad.norm().item():.4f} | "
            f"q_grad_norm={q_grad.norm().item():.4f}"
        )
