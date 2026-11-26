"""PennyLane Variational Quantum Circuit (VQC) layer wrapped in qml.qnn.TorchLayer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
import torch
import torch.nn as nn


def _make_device(device_name: str, n_qubits: int) -> qml.Device:
    """Create a PennyLane device, falling back to default.qubit if lightning is unavailable."""
    try:
        return qml.device(device_name, wires=n_qubits)
    except Exception:
        return qml.device("default.qubit", wires=n_qubits)


def build_qnode(
    n_qubits: int = 4,
    n_layers: int = 3,
    encoding_type: Literal["angle", "reuploading"] = "angle",
    device_name: str = "lightning.qubit",
):
    """Construct a PennyLane QNode for either standard angle encoding or data re-uploading."""
    dev = _make_device(device_name, n_qubits)
    if dev.name == "default.qubit":
        diff_method = "backprop"
    elif dev.name == "lightning.qubit":
        diff_method = "adjoint"
    else:
        diff_method = "best"
    enc = encoding_type.lower()
    if enc not in ("angle", "reuploading"):
        raise ValueError(f"Unsupported encoding_type {encoding_type!r}. Choose 'angle' or 'reuploading'.")

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        if enc == "angle":
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        else:
            for layer_idx in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer_idx : layer_idx + 1], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class QuantumReadoutLayer(nn.Module):
    """Differentiable PennyLane VQC layer compatible with CPU/GPU PyTorch training."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        encoding_type: Literal["angle", "reuploading"] = "angle",
        device_name: str = "lightning.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        self.device_name = device_name

        qnode = build_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_type=encoding_type,
            device_name=device_name,
        )
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run quantum circuit on CPU (matching lightning.qubit) and return on x.device."""
        orig_device = x.device
        # Keep TorchLayer evaluation on CPU to avoid CUDA/CPU mismatches with lightning.qubit,
        # while preserving autograd backprop to the upstream CUDA/CPU encoder.
        if orig_device.type != "cpu":
            x_cpu = x.to("cpu")
            # Ensure TorchLayer weights are on CPU during QNode execution
            if next(self.q_layer.parameters()).device.type != "cpu":
                self.q_layer.to("cpu")
            out_cpu = self.q_layer(x_cpu)
            return out_cpu.to(orig_device)
        return self.q_layer(x)


def draw_and_save_circuit(
    n_qubits: int = 4,
    n_layers: int = 3,
    out_dir: str | Path = "results/figures",
) -> list[Path]:
    """Export clean circuit diagrams for both 'angle' and 'reuploading' ansatze."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    sample_inputs = torch.zeros(n_qubits)
    sample_weights = torch.zeros((n_layers, n_qubits, 3))

    for enc in ("angle", "reuploading"):
        qnode = build_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding_type=enc,
            device_name="default.qubit",
        )
        fig, _ = qml.draw_mpl(qnode, level="device")(sample_inputs, sample_weights)
        fig.suptitle(
            f"Hybrid QGNN Circuit ({enc.capitalize()} Encoding, {n_qubits}q x {n_layers}L)",
            fontsize=11,
            y=1.02,
        )
        file_path = out_path / f"circuit_{enc}_{n_qubits}q{n_layers}l.png"
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(file_path)

    return saved_files


if __name__ == "__main__":
    paths = draw_and_save_circuit()
    for p in paths:
        print(f"Saved circuit diagram: {p}")
