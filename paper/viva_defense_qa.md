# Viva Defense Q&A: 12 Toughest Examiner Questions

### 1. Are you claiming "quantum advantage" or "quantum supremacy" with 4 qubits?
**Answer:** No. A 4-qubit circuit is trivially simulable classically and does not offer asymptotic speedup. Our contribution is strictly scientific and methodological: under an **exact 41-parameter head parity** (`36` circuit weights + `5` linear readout weights), we evaluate whether a variational quantum readout matches a parameter-matched classical MLP on strict out-of-distribution Bemis–Murcko scaffold splits (`0.8340` vs. `0.8360` ROC-AUC on BACE, with higher PR-AUC `0.7455` vs. `0.7168`), and we characterize its behavior under real NISQ hardware noise (`ibm_fez`).

### 2. Why did you use 4 qubits instead of 50+ qubits?
**Answer:** First, encoding all atoms of a 30-atom drug directly into 30+ qubits is infeasible on NISQ hardware due to gate error accumulation and barren plateaus. Second, using a classical `GINEConv` encoder (`42,500` params) to compress molecular topology into a 4-dimensional $[-\pi, \pi]$ latent space keeps our circuit shallow (`4` qubits, `12` CNOTs, ideal depth `14`, transpiled `ibm_fez` depth `71` with `21` two-qubit gates), enabling high-fidelity execution (`0.8981` vs. `0.9002` ROC-AUC). In Ablation C, increasing to `6` qubits ($L=3$, `18` CNOTs) yielded `0.8400 ± 0.0024` ROC-AUC, while `4q × 1L` (`4` CNOTs) achieved `0.8479 ± 0.0134`.

### 3. Your classical GNN encoder has 42,500 parameters while the quantum head has only 41. Isn't the GNN doing all the work?
**Answer:** The GNN encoder extracts structural representations, which is why **both** `classical_matched` and `hybrid_angle` share the exact same `42,500`-parameter `GINEConv` architecture and `41`-parameter head budget. Any performance or calibration difference between `classical_matched` and `hybrid_angle` is therefore isolated to the 41-parameter readout head rather than encoder capacity.

### 4. Why did `hybrid_angle` perform competitively on BACE (`0.8340`) but trail `classical_matched` on BBBP (`0.6928` vs. `0.7479`)?
**Answer:** Two factors explain this dataset-dependent trade-off: (1) **Class & scaffold skew:** BBBP's test split has `88.73%` positives across `204` singleton test scaffolds, making the decision boundary sensitive to calibration; (2) **Inductive bias:** `hybrid_angle` actually achieved the highest Test F1 on BBBP (`0.9103 ± 0.0185`), whereas its periodic $\sin/\cos$ Bloch-sphere decision boundary struggled more on BBBP's OOD ranking ($p=0.0076$) than on BACE's balanced binding-site task (`48.4%` train positives, $p=0.8711$).

### 5. Why did `hybrid_reuploading` perform worse and show higher variance than `hybrid_angle`?
**Answer:** Data re-uploading applies $R_Y(\mathbf{z})$ before all `3` entangling layers, creating higher-frequency Fourier modes in the latent variables $\mathbf{z}$. With finite training data (`1,210`–`1,631` molecules), those higher-frequency periodic landscapes are harder to optimize and more prone to seed-to-seed local minima (e.g., BACE `seed=46` dropped to `0.6018` ROC-AUC, driving std to `0.1045`).

### 6. Why did you enforce Bemis–Murcko scaffold splitting instead of standard k-fold cross-validation?
**Answer:** Random splitting (or standard k-fold) leaks near-identical molecular analogues across train and test sets. In Ablation A, switching from scaffold split to random split artificially inflated `hybrid_angle` BBBP ROC-AUC from `0.6928` to `0.8848` (`+0.1921`) and `classical_matched` from `0.7479` to `0.9061` (`+0.1581`). Bemis–Murcko splitting guarantees `train_scaffolds ∩ test_scaffolds = ∅`, measuring true OOD drug-discovery generalization.

### 7. How do you avoid barren plateaus in your quantum circuit?
**Answer:** Barren plateaus emerge exponentially with qubit count $n_q$ and circuit depth $L$ under global cost functions. We mitigate them by: (1) restricting $n_q \in \{4, 6\}$ and $L \in \{1, 2, 3\}$; (2) measuring **local** single-qubit Pauli-$Z$ observables $\langle Z_i \rangle$ rather than a global $n_q$-qubit projector; and (3) verifying non-vanishing quantum weight gradients during unit testing (`grad_norm = 0.1039` for `angle`, `0.1606` for `reuploading`).

### 8. Why did Test ROC-AUC in `noise_sim.py` stay around `0.8605–0.8642` even at `p_depol = 0.05`?
**Answer:** Depolarizing noise contracts the density matrix toward the maximally mixed state $I/2^{n_q}$, isotropically shrinking $\langle Z_i \rangle$ toward `0` (hence MAE rises to `0.3075` at $p=0.05$). Because ROC-AUC is strictly rank-invariant to positive linear scaling of logits ($\mathbf{w}^\top ((1-\lambda)\hat{\mathbf{e}}) + b$), the relative ordering of molecules is preserved even as raw probability calibration degrades.

### 9. How does your Zero-Noise Extrapolation (ZNE) work mathematically, and did it help?
**Answer:** We use digital unitary folding at scale factors $s=1$ ($U$) and $s=3$ ($U U^\dagger U$) on `default.mixed`, then apply linear Richardson extrapolation $\hat{e}_{\text{ZNE}} = \operatorname{clip}(1.5\,\hat{e}^{(1)} - 0.5\,\hat{e}^{(3)}, -1, 1)$. It reduced Pauli-$Z$ expectation MAE at every non-zero noise level—for instance, by **71%** at $p=0.005$ (`0.0624 → 0.0182`) and **60%** at $p=0.010$ (`0.1093 → 0.0441`).

### 10. How did you verify that your Qiskit circuit on `ibm_fez` was identical to your trained PennyLane circuit?
**Answer:** Before calling IBM Cloud, `ibm_qpu_eval.py` runs a local Qiskit `Statevector` equivalence check across all `50` stratified test molecules, accounting for Qiskit's little-endian Pauli string ordering (`"I"*(n-1-i) + "Z" + "I"*i`) and `qml.Rot(phi,theta,omega)` = `RZ(omega) RY(theta) RZ(phi)` decomposition. The maximum absolute error against PennyLane's ideal simulator was `1.266e-07` (well below our `1e-5` threshold).

### 11. Why did your 12-CNOT ideal circuit become depth `71` with `21` two-qubit gates on `ibm_fez`?
**Answer:** Ideal PennyLane assumes all-to-all qubit connectivity and native `CNOT` gates. On `ibm_fez` (heavy-hex topology with native `CZ`/`ECR` basis gates), the Qiskit transpiler (`optimization_level=3`) inserts `SWAP` routing operations and decomposes `Rot` + `CNOT` into native basis gates, increasing depth from `14` to `71` and two-qubit gates from `12` to `21`. Even at depth `71`, `ibm_fez` achieved `0.8981` ROC-AUC vs. `0.9002` ideal.

### 12. Why did `hybrid_angle` at `25%` BACE training data (`0.7228`) drop more than `classical_matched` (`0.8025`)?
**Answer:** With only `25%` of training scaffolds ($\approx 302$ molecules), the 4-dimensional periodic trigonometric landscape of $R_Y(\mathbf{z})$ + `StronglyEntanglingLayers` has fewer gradient updates per epoch to align the `GINEConv` bottleneck with the Bloch sphere, whereas `ReLU` MLPs optimize piecewise-linearly. At `50%` data ($\approx 605$ molecules), `hybrid_angle` recovers completely to `0.8370 ± 0.0251`, slightly ahead of `classical_matched` (`0.8236 ± 0.0192`).
