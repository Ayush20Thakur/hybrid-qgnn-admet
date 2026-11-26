"""Interactive Streamlit Dashboard: Hybrid QGNN — Quantum-Enhanced Drug Screening."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.explainability.substructure_viz import explain_molecule, load_trained_model

st.set_page_config(
    page_title="Hybrid QGNN — Quantum-Enhanced Drug Screening",
    page_icon="⚛️",
    layout="wide",
)

PRESET_DRUGS: dict[str, str] = {
    "Caffeine (CNS Permeable)": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "L-DOPA (CNS Prodrug)": "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",
    "Aspirin (Low BBB / NSAID)": "CC(=O)Oc1ccccc1C(=O)O",
    "Dopamine (Non-Permeable Neurotransmitter)": "NCCc1ccc(O)c(O)c1",
    "Atenolol (Non-CNS Beta-Blocker)": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "BACE-1 Inhibitor (Scaffold Test Active)": "Fc1cc(cc(F)c1)CC(NC(=O)c1cc(N(C)S(=O)(=O)C)cc(c1)C(=O)N[C@H](C)c1ccccc1)C(O)C[NH2+]Cc1cccc(C(C)(C)C)c1",
}


@st.cache_resource
def get_cached_model(dataset_key: str):
    return load_trained_model(dataset_key, "hybrid_angle")


def main() -> None:
    st.title("⚛️ Hybrid QGNN — Quantum-Enhanced Drug Screening")
    st.caption(
        "Noise-Aware Hybrid Quantum-Classical Graph Neural Networks under Strict Bemis–Murcko Scaffold Splitting"
    )

    tab_screener, tab_benchmarks = st.tabs(
        ["🔬 Live Molecule Screener", "📊 Research Benchmarks & IBM QPU Validation"]
    )

    with tab_screener:
        col_left, col_right = st.columns([1.05, 1.35], gap="large")

        with col_left:
            task_choice = st.radio(
                "Screening Endpoint",
                options=["BACE-1 Inhibition (Alzheimer's)", "BBBP Permeability (CNS)"],
                horizontal=True,
            )
            ds_key = "bace" if task_choice.startswith("BACE") else "bbbp"

            preset_name = st.selectbox("Preset Drug / Test Molecule", list(PRESET_DRUGS.keys()))
            smiles_input = st.text_input("SMILES String", value=PRESET_DRUGS[preset_name])

            model, ckpt_path = get_cached_model(ds_key)
            if model is None:
                st.error(f"No trained checkpoint found for {ds_key.upper()} hybrid_angle.")
                return

            try:
                res = explain_molecule(smiles_input, model, width=500, height=330)
            except ValueError as exc:
                st.error(f"Invalid SMILES: {exc}")
                return

            prob = res["probability"]
            st.metric(
                label=f"Hybrid QGNN Prediction — {task_choice}",
                value=f"{prob * 100:.1f}%",
                delta="Active / Permeable" if prob >= 0.5 else "Inactive / Non-Permeable",
            )
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Checkpoint: `{ckpt_path}`")

            props = res["properties"]
            st.markdown(f"**Bemis–Murcko Scaffold:** `{props['scaffold']}`")
            p_cols = st.columns(5)
            p_cols[0].metric("MW", f"{props['MW']}")
            p_cols[1].metric("LogP", f"{props['LogP']}")
            p_cols[2].metric("TPSA", f"{props['TPSA']}")
            p_cols[3].metric("HBD", f"{props['HBD']}")
            p_cols[4].metric("HBA", f"{props['HBA']}")

        with col_right:
            st.subheader("Atom-Level Gradient Saliency")
            st.image(res["png_bytes"], caption="Highlighted atoms contribute most to the Hybrid QGNN logit.")

            with st.expander("⚛️ Quantum Circuit Inspector (4-Qubit × 3-Layer VQC)", expanded=True):
                q_df = pd.DataFrame(
                    {
                        "Qubit": [f"q[{i}]" for i in range(4)],
                        "Encoded Angle θ (rad)": [round(a, 4) for a in res["latent_angles"]],
                        "Pauli-Z ⟨Z⟩": [round(z, 4) for z in res["pauli_z_expvals"]],
                    }
                )
                st.dataframe(q_df, use_container_width=True, hide_index=True)

                circuit_img = ROOT_DIR / "results/figures/circuit_angle_4q3l.png"
                if circuit_img.exists():
                    st.image(str(circuit_img), caption="4-Qubit AngleEncoding + StronglyEntanglingLayers Circuit")

    with tab_benchmarks:
        st.subheader("Real IBM Quantum QPU Validation (`ibm_fez`)")
        qpu_json = ROOT_DIR / "results/metrics/ibm_qpu_validation.json"
        if qpu_json.exists():
            with qpu_json.open("r", encoding="utf-8") as f:
                qpu = json.load(f)
            m_cols = st.columns(6)
            m_cols[0].metric("IBM Backend", qpu["backend"])
            m_cols[1].metric("Job ID", qpu["job_id"])
            m_cols[2].metric("Ideal ROC-AUC", f"{qpu['ideal_roc_auc']:.4f}")
            m_cols[3].metric("QPU ROC-AUC", f"{qpu['qpu_roc_auc']:.4f}")
            m_cols[4].metric("MAE ⟨Z⟩", f"{qpu['mae_expectation']:.4f}")
            m_cols[5].metric("Depth / 2Q Gates", f"{qpu['transpiled_depth']} / {qpu['two_qubit_gate_count']}")

        summary_csv = ROOT_DIR / "results/metrics/summary_main.csv"
        if summary_csv.exists():
            st.subheader("5-Seed Scaffold-Split Benchmark Summary")
            st.dataframe(pd.read_csv(summary_csv), use_container_width=True, hide_index=True)

        fig_specs = [
            ("IBM QPU vs. Ideal Simulator Parity", "results/figures/ibm_qpu_parity.png"),
            ("NISQ Noise Sweep & ZNE Mitigation", "results/figures/nisq_noise_zne.png"),
            ("Ablation A: Random vs. Scaffold Split Leakage", "results/figures/ablation_split_leakage.png"),
            ("Ablation B: Low-Data Pharmaceutical Regime", "results/figures/ablation_low_data.png"),
            ("Ablation C: Qubit & Depth Sweep (BACE)", "results/figures/ablation_qubit_depth.png"),
        ]
        st.subheader("Publication-Grade Figures (300 DPI)")
        cols = st.columns(2)
        for idx, (title, rel_path) in enumerate(fig_specs):
            img_path = ROOT_DIR / rel_path
            if img_path.exists():
                cols[idx % 2].image(str(img_path), caption=title, use_container_width=True)


if __name__ == "__main__":
    main()
