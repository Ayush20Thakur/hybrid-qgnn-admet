"""Real IBM Quantum QPU validation on a stratified 50-molecule BACE test subset."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import seaborn as sns
import torch
from dotenv import load_dotenv
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.quantum.noise_sim import find_best_bace_checkpoint, load_frozen_bace_model

warnings.filterwarnings("ignore", message="The usage of `scatter\\(reduce='max'\\)`")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def build_parameterized_qiskit_circuit(
    weights: np.ndarray,
    n_qubits: int = 4,
) -> tuple[QuantumCircuit, ParameterVector]:
    """Reconstruct AngleEmbedding(Y) + StronglyEntanglingLayers as a parameterized Qiskit circuit."""
    x_params = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.ry(x_params[i], i)

    n_layers = weights.shape[0]
    for l_idx in range(n_layers):
        r = (l_idx % (n_qubits - 1)) + 1
        for i in range(n_qubits):
            phi, theta, omega = [float(v) for v in weights[l_idx, i]]
            qc.rz(phi, i)
            qc.ry(theta, i)
            qc.rz(omega, i)
        for i in range(n_qubits):
            qc.cx(i, (i + r) % n_qubits)

    return qc, x_params


def build_pauli_z_observables(n_qubits: int = 4) -> list[SparsePauliOp]:
    """Build single-qubit Pauli-Z observables in Qiskit little-endian qubit ordering."""
    obs_list = []
    for i in range(n_qubits):
        pauli_str = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        obs_list.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
    return obs_list


def verify_local_equivalence(
    qc: QuantumCircuit,
    x_params: ParameterVector,
    observables: list[SparsePauliOp],
    x_subset: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Assert Qiskit Statevector matches PennyLane ideal Pauli-Z expectations (< 1e-5)."""
    n_qubits = x_subset.shape[1]
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def pl_ideal(inputs):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    pl_exps = np.array([pl_ideal(x) for x in x_subset], dtype=float)

    qk_exps = np.zeros_like(pl_exps)
    for row_idx, x_vals in enumerate(x_subset):
        bound_qc = qc.assign_parameters({x_params[i]: float(x_vals[i]) for i in range(n_qubits)})
        sv = Statevector.from_instruction(bound_qc)
        for q_idx, obs in enumerate(observables):
            qk_exps[row_idx, q_idx] = float(np.real(sv.expectation_value(obs)))

    max_err = float(np.max(np.abs(pl_exps - qk_exps)))
    print(f"Local Statevector equivalence check: max_abs_err = {max_err:.3e}")
    assert max_err < 1e-5, f"Equivalence check failed: max_err={max_err}"
    return pl_exps


def main() -> None:
    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")
    crn = os.getenv("IBM_QUANTUM_CRN")
    if not token or not crn:
        raise RuntimeError("IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN must be set in .env")

    ckpt_path = find_best_bace_checkpoint()
    model, x_latent, y_true, q_weights = load_frozen_bace_model(ckpt_path)

    # Stratified 50-molecule subset preserving BACE test class ratio
    idx_all = np.arange(len(y_true))
    sub_idx, _ = train_test_split(
        idx_all, train_size=50, stratify=y_true.astype(int), random_state=42
    )
    x_sub = x_latent[sub_idx]
    y_sub = y_true[sub_idx]
    print(f"Stratified subset: N={len(y_sub)} | positives={int(y_sub.sum())} ({y_sub.mean():.3f})")

    qc, x_params = build_parameterized_qiskit_circuit(q_weights, n_qubits=4)
    observables = build_pauli_z_observables(n_qubits=4)
    ideal_exps = verify_local_equivalence(qc, x_params, observables, x_sub, q_weights)

    # Connect to IBM Quantum Cloud & select least busy real QPU
    service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=crn)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=4)
    print(f"Selected IBM QPU backend: {backend.name}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(qc)
    isa_observables = [obs.apply_layout(isa_circuit.layout) for obs in observables]

    transpiled_depth = int(isa_circuit.depth())
    ops_count = isa_circuit.count_ops()
    two_qubit_gates = int(
        sum(cnt for gate, cnt in ops_count.items() if gate in ("ecr", "cz", "cx", "rzz"))
    )
    print(f"Transpiled depth={transpiled_depth}, 2-qubit gates={two_qubit_gates}")

    # Single batched EstimatorV2 job (50 PUBs x 4 Pauli-Z observables, 1024 shots)
    pubs = [(isa_circuit, isa_observables, [float(v) for v in row]) for row in x_sub]
    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = 1024

    print("Submitting batched EstimatorV2 job to IBM Quantum...")
    job = estimator.run(pubs)
    job_id = job.job_id()
    print(f"Submitted IBM QPU Job ID: {job_id} — waiting for results...")
    result = job.result()

    qpu_exps = np.array([np.asarray(res.data.evs, dtype=float) for res in result])

    with torch.no_grad():
        ideal_probs = torch.sigmoid(
            model.classifier(torch.tensor(ideal_exps, dtype=torch.float32))
        ).view(-1).numpy()
        qpu_probs = torch.sigmoid(
            model.classifier(torch.tensor(qpu_exps, dtype=torch.float32))
        ).view(-1).numpy()

    ideal_roc = float(roc_auc_score(y_sub, ideal_probs))
    ideal_pr = float(average_precision_score(y_sub, ideal_probs))
    qpu_roc = float(roc_auc_score(y_sub, qpu_probs))
    qpu_pr = float(average_precision_score(y_sub, qpu_probs))
    mae_exp = float(np.mean(np.abs(qpu_exps - ideal_exps)))
    mae_prob = float(np.mean(np.abs(qpu_probs - ideal_probs)))

    summary = {
        "dataset": "bace",
        "checkpoint": str(ckpt_path),
        "n_molecules": int(len(y_sub)),
        "n_positives": int(y_sub.sum()),
        "backend": backend.name,
        "job_id": job_id,
        "shots": 1024,
        "optimization_level": 3,
        "transpiled_depth": transpiled_depth,
        "two_qubit_gate_count": two_qubit_gates,
        "ideal_roc_auc": ideal_roc,
        "ideal_pr_auc": ideal_pr,
        "qpu_roc_auc": qpu_roc,
        "qpu_pr_auc": qpu_pr,
        "mae_expectation": mae_exp,
        "mae_probability": mae_prob,
    }

    out_json = Path("results/metrics/ibm_qpu_validation.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 300-DPI parity scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    axes[0].scatter(ideal_exps.ravel(), qpu_exps.ravel(), alpha=0.65, s=22, color="#2b5c8f", edgecolor="none")
    axes[0].plot([-1, 1], [-1, 1], "k--", lw=1.2)
    axes[0].set_title(f"Pauli-Z Expectations ({backend.name})")
    axes[0].set_xlabel("Ideal Simulator ⟨Z⟩")
    axes[0].set_ylabel("Real IBM QPU ⟨Z⟩")
    axes[0].text(0.05, 0.90, f"MAE = {mae_exp:.4f}", transform=axes[0].transAxes, fontsize=9.5)

    axes[1].scatter(ideal_probs, qpu_probs, alpha=0.75, s=30, color="#1b9e77", edgecolor="none")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1.2)
    axes[1].set_title("BACE Predicted Probabilities")
    axes[1].set_xlabel("Ideal Simulator P(Active)")
    axes[1].set_ylabel("Real IBM QPU P(Active)")
    axes[1].text(
        0.05,
        0.84,
        f"Ideal AUC = {ideal_roc:.4f}\nQPU AUC   = {qpu_roc:.4f}",
        transform=axes[1].transAxes,
        fontsize=9.5,
    )

    fig.tight_layout()
    out_fig = Path("results/figures/ibm_qpu_parity.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n=== REAL IBM QPU VALIDATION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print(f"\nSaved: {out_json} and {out_fig}")


if __name__ == "__main__":
    main()
