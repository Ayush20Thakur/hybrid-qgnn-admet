# Hybrid Quantum-Classical Graph Neural Network (QGNN) for ADMET Drug Discovery

**Noise-Aware Hybrid Quantum-Classical Graph Neural Networks for Out-of-Distribution Molecular Property & ADMET Prediction Under Bemis–Murcko Scaffold Splitting**

---

## Overview

This repository implements an end-to-end **Hybrid Quantum-Classical Graph Neural Network (QGNN)** for molecular property prediction on **MoleculeNet BBBP** ($N=2{,}039$) and **BACE** ($N=1{,}513$), designed to address three methodological gaps in quantum cheminformatics:

1. **Leak-Free OOD Evaluation:** Strict deterministic 80/10/10 **Bemis–Murcko scaffold splitting** via RDKit (`train_scaffolds ∩ val_scaffolds ∩ test_scaffolds = ∅`), audited against random splitting.
2. **Exact Parameter-Matched Baselines:** A 4-qubit, 3-layer PennyLane VQC readout (`36` circuit weights + `5` linear readout = **`41` head parameters**, `42,541` total) benchmarked against a **`41`-head-parameter** classical MLP (`ClassicalGNNParamMatched`) and a `193`-head-parameter MLP (`ClassicalGNNFull`).
3. **NISQ Noise Mitigation & Real IBM QPU Validation:** Density-matrix (`default.mixed`) depolarizing + readout noise sweeps with **Zero-Noise Extrapolation (ZNE)** and real-hardware execution on IBM Quantum (`ibm_fez`).

---

## Architecture

- **Graph Featurizer (`src/data/featurizer.py`):** Converts SMILES to PyTorch Geometric `Data` graphs with `45`-dim atom node features and `12`-dim bond edge features.
- **Classical GNN Encoder (`src/models/gnn_encoder.py`):** 3-layer residual `GINEConv` with `BatchNorm1d`, `ReLU`, `Dropout(0.2)`, dual `global_mean_pool + global_max_pool`, and a bottleneck projecting to $[-\pi, \pi]^{n_q}$ (`42,500` params).
- **Quantum Readout (`src/models/quantum_layer.py`, `src/models/hybrid_qgnn.py`):** PennyLane `qml.qnn.TorchLayer` supporting `angle` (`AngleEmbedding(Y)` + `StronglyEntanglingLayers`) and `reuploading` (data re-uploading) ansätze, measuring $\langle Z_i \rangle$ on each qubit.
- **Explainability & Dashboard (`src/explainability/substructure_viz.py`, `app/streamlit_app.py`):** Atom-level gradient saliency ($\|\partial\,\text{logit}/\partial\,\mathbf{x}_v\|_2$) rendered with RDKit + interactive Streamlit screening app.

---

## Benchmark Results

### 1. 5-Seed Scaffold-Split Benchmark (Seeds `42–46`)

| Dataset | Model | Head / Total Params | Test ROC-AUC | Test PR-AUC | Test F1 |
| :--- | :--- | ---: | :---: | :---: | :---: |
| **BBBP** | `classical_full` | 193 / 42,693 | 0.7547 ± 0.0535 | 0.9621 ± 0.0109 | 0.8840 ± 0.0531 |
| **BBBP** | `classical_matched` | 41 / 42,541 | 0.7479 ± 0.0434 | 0.9595 ± 0.0120 | 0.9007 ± 0.0378 |
| **BBBP** | `hybrid_angle` | 41 / 42,541 | 0.6928 ± 0.0235 | 0.9316 ± 0.0153 | **0.9103 ± 0.0185** |
| **BBBP** | `hybrid_reuploading` | 41 / 42,541 | 0.6735 ± 0.0666 | 0.9381 ± 0.0169 | 0.8718 ± 0.0411 |
| **BACE** | `classical_full` | 193 / 42,693 | 0.8410 ± 0.0119 | 0.7286 ± 0.0293 | 0.6493 ± 0.0388 |
| **BACE** | `classical_matched` | 41 / 42,541 | 0.8360 ± 0.0031 | 0.7168 ± 0.0267 | 0.6566 ± 0.0282 |
| **BACE** | `hybrid_angle` | 41 / 42,541 | 0.8340 ± 0.0280 | **0.7455 ± 0.0275** | 0.6563 ± 0.0306 |
| **BACE** | `hybrid_reuploading` | 41 / 42,541 | 0.7800 ± 0.1045 | 0.6689 ± 0.1329 | 0.5960 ± 0.1279 |

### 2. Real IBM Quantum QPU Validation (`ibm_fez`, Stratified $N=50$ BACE Test Subset)

| Metric | Ideal Simulator | Real IBM QPU (`ibm_fez`) |
| :--- | :---: | :---: |
| **Job ID** | — | `dc3q0pls2omc738e1m2g` |
| **Shots / Opt. Level** | Statevector | 1,024 / Level 3 |
| **Transpiled Depth / 2Q Gates** | 14 / 12 | 71 / 21 |
| **Test ROC-AUC** | 0.9002 | **0.8981** |
| **Test PR-AUC** | 0.7348 | **0.7193** |
| **MAE ($\langle Z \rangle$ / Prob.)** | 0.0000 | 0.0677 / 0.0297 |

---

## Setup & Usage

### 1. Environment Setup
```powershell
uv venv .venv --python 3.12 --seed
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.\.venv\Scripts\python.exe verify_env.py
```

### 2. Data Preparation & Scaffold Splitting
```powershell
.\.venv\Scripts\python.exe -m src.data.scaffold_split
```

### 3. Training & Ablations
```powershell
# Classical + Hybrid 5-seed scaffold benchmark
.\.venv\Scripts\python.exe -u -m src.training.train --datasets bbbp bace --models classical_full classical_matched hybrid_angle hybrid_reuploading --split scaffold --quantum_device default.qubit --compare_all

# Aggregate tables, p-values, and 300-DPI figures
.\.venv\Scripts\python.exe -m src.training.evaluate
```

### 4. NISQ Noise Sweep & IBM QPU Validation
```powershell
.\.venv\Scripts\python.exe -m src.quantum.noise_sim
.\.venv\Scripts\python.exe -m src.quantum.ibm_qpu_eval   # Requires IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN in .env
```

### 5. Launch Interactive Drug Screening App
```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```
