"""Environment verification script for Hybrid QGNN ADMET project (Phase 0)."""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Callable


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_pytorch() -> str:
    import torch

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        dev_name = torch.cuda.get_device_name(0)
        status = f"torch {torch.__version__} | CUDA available ({dev_name})"
    else:
        status = f"torch {torch.__version__} | CPU only"
    x = torch.randn(2, 4)
    y = x @ x.T
    assert y.shape == (2, 2)
    return status


def check_rdkit() -> str:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    # Ibuprofen SMILES
    smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, "Failed to parse Ibuprofen SMILES"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    assert scaffold == "c1ccccc1", f"Unexpected scaffold: {scaffold}"
    return f"rdkit {rdkit.__version__} | atoms={mol.GetNumAtoms()}, scaffold='{scaffold}'"


def check_pyg() -> str:
    import torch_geometric
    from torch_geometric.datasets import MoleculeNet  # noqa: F401
    from torch_geometric.nn import GINConv, GATv2Conv, global_mean_pool  # noqa: F401

    return f"torch_geometric {torch_geometric.__version__} | GINConv/GATv2Conv/MoleculeNet OK"


def check_pennylane_torchlayer() -> str:
    import pennylane as qml
    import torch
    import torch.nn as nn

    n_qubits = 4
    n_layers = 2
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    model = nn.Sequential(qlayer, nn.Linear(n_qubits, 1))

    x = torch.randn(3, n_qubits, requires_grad=True)
    out = model(x)
    assert out.shape == (3, 1), f"Unexpected output shape: {out.shape}"
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all(), "Input gradients missing"
    q_weights = next(qlayer.parameters())
    assert q_weights.grad is not None and torch.isfinite(q_weights.grad).all(), (
        "Quantum weight gradients missing"
    )
    return (
        f"pennylane {qml.__version__} (lightning.qubit) | "
        f"forward={tuple(out.shape)}, grad_norm={q_weights.grad.norm().item():.4f}"
    )


def check_qiskit() -> str:
    import qiskit
    import qiskit_ibm_runtime

    return f"qiskit {qiskit.__version__} | qiskit-ibm-runtime {qiskit_ibm_runtime.__version__}"


def main() -> int:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("(a) PyTorch + GPU/CPU", check_pytorch),
        ("(b) RDKit SMILES + Bemis-Murcko", check_rdkit),
        ("(c) PyTorch Geometric", check_pyg),
        ("(d) PennyLane lightning.qubit TorchLayer", check_pennylane_torchlayer),
        ("(e) Qiskit + IBM Runtime", check_qiskit),
    ]

    results: list[CheckResult] = []
    for name, fn in checks:
        try:
            detail = fn()
            results.append(CheckResult(name=name, passed=True, detail=detail))
        except Exception as exc:
            traceback.print_exc()
            results.append(CheckResult(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}"))

    print("\n" + "=" * 92)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print("-" * 92)
    print(f"{'Check':<40} | {'Status':<6} | {'Details'}")
    print("-" * 92)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:<40} | {status:<6} | {r.detail}")
    print("=" * 92)

    all_passed = all(r.passed for r in results)
    print(f"Summary: {sum(r.passed for r in results)}/{len(results)} checks passed.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
