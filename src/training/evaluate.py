"""Aggregate Phase 2-4 JSON metrics, run statistical tests, and export tables & 300-DPI plots."""

from __future__ import annotations

import json
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
from scipy import stats

from src.models.quantum_layer import build_qnode

METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def load_all_runs(metrics_dir: Path = METRICS_DIR) -> list[dict[str, Any]]:
    """Load all per-seed experiment JSONs from results/metrics/."""
    runs: list[dict[str, Any]] = []
    for p in sorted(metrics_dir.glob("*.json")):
        if p.name.startswith("ibm_"):
            continue
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "test" in data and "model" in data:
            runs.append(data)
    return runs


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def compute_paired_tests(
    runs: list[dict[str, Any]],
    dataset: str,
    model_a: str = "hybrid_angle",
    model_b: str = "classical_matched",
    split: str = "scaffold",
    train_fraction: float = 1.0,
    metric: str = "roc_auc",
) -> dict[str, Any]:
    """Compute paired t-test and Wilcoxon signed-rank test on matched seeds."""
    a_map = {
        r["seed"]: r["test"][metric]
        for r in runs
        if r["dataset"] == dataset
        and r["model"] == model_a
        and r["split"] == split
        and np.isclose(r["train_fraction"], train_fraction)
        and r.get("n_qubits", 4) == 4
        and r.get("n_layers", 3) == 3
    }
    b_map = {
        r["seed"]: r["test"][metric]
        for r in runs
        if r["dataset"] == dataset
        and r["model"] == model_b
        and r["split"] == split
        and np.isclose(r["train_fraction"], train_fraction)
        and r.get("n_qubits", 4) == 4
        and r.get("n_layers", 3) == 3
    }
    common_seeds = sorted(set(a_map) & set(b_map))
    if len(common_seeds) < 2:
        return {"n_seeds": len(common_seeds), "t_stat": float("nan"), "t_pval": float("nan"), "w_pval": float("nan")}

    a_vals = np.array([a_map[s] for s in common_seeds])
    b_vals = np.array([b_map[s] for s in common_seeds])
    t_res = stats.ttest_rel(a_vals, b_vals)
    try:
        w_res = stats.wilcoxon(a_vals, b_vals)
        w_pval = float(w_res.pvalue)
    except ValueError:
        w_pval = float("nan")
    return {
        "n_seeds": len(common_seeds),
        "mean_diff": float(np.mean(a_vals - b_vals)),
        "t_stat": float(t_res.statistic),
        "t_pval": float(t_res.pvalue),
        "w_pval": w_pval,
    }


def get_circuit_specs(n_qubits: int, n_layers: int) -> tuple[int, int]:
    """Return (cnot_count, circuit_depth) for StronglyEntanglingLayers angle circuit."""
    qnode = build_qnode(n_qubits=n_qubits, n_layers=n_layers, encoding_type="angle", device_name="default.qubit")
    specs = qml.specs(qnode, level="device")(torch.zeros(n_qubits), torch.zeros((n_layers, n_qubits, 3)))
    res = specs["resources"]
    cnot_count = int(res.gate_types.get("CNOT", 0))
    depth = int(res.depth)
    return cnot_count, depth


def build_main_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate Phase 2 & 3 full scaffold results across all 4 models."""
    rows = []
    for ds in ("bbbp", "bace"):
        for mdl in ("classical_full", "classical_matched", "hybrid_angle", "hybrid_reuploading"):
            sub = [
                r for r in runs
                if r["dataset"] == ds and r["model"] == mdl and r["split"] == "scaffold"
                and np.isclose(r["train_fraction"], 1.0)
                and r.get("n_qubits", 4) == 4 and r.get("n_layers", 3) == 3
            ]
            if not sub:
                continue
            auc_m, auc_s = _mean_std([r["test"]["roc_auc"] for r in sub])
            pr_m, pr_s = _mean_std([r["test"]["pr_auc"] for r in sub])
            f1_m, f1_s = _mean_std([r["test"]["f1"] for r in sub])
            rows.append({
                "Dataset": ds.upper(),
                "Model": mdl,
                "Seeds": len(sub),
                "Head_Params": sub[0]["params"]["head"],
                "Total_Params": sub[0]["params"]["total"],
                "ROC_AUC_Mean": auc_m,
                "ROC_AUC_Std": auc_s,
                "PR_AUC_Mean": pr_m,
                "PR_AUC_Std": pr_s,
                "F1_Mean": f1_m,
                "F1_Std": f1_s,
            })
    return pd.DataFrame(rows)


def build_split_leakage_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Ablation A: Random vs. Bemis-Murcko Scaffold split."""
    rows = []
    for ds in ("bbbp", "bace"):
        for mdl in ("classical_matched", "hybrid_angle"):
            for sp in ("scaffold", "random"):
                sub = [
                    r for r in runs
                    if r["dataset"] == ds and r["model"] == mdl and r["split"] == sp
                    and np.isclose(r["train_fraction"], 1.0)
                    and r.get("n_qubits", 4) == 4 and r.get("n_layers", 3) == 3
                ]
                if not sub:
                    continue
                auc_m, auc_s = _mean_std([r["test"]["roc_auc"] for r in sub])
                pr_m, pr_s = _mean_std([r["test"]["pr_auc"] for r in sub])
                rows.append({
                    "Dataset": ds.upper(),
                    "Model": mdl,
                    "Split": sp,
                    "Seeds": len(sub),
                    "ROC_AUC_Mean": auc_m,
                    "ROC_AUC_Std": auc_s,
                    "PR_AUC_Mean": pr_m,
                    "PR_AUC_Std": pr_s,
                })
    return pd.DataFrame(rows)


def build_low_data_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Ablation B: Low-data training fraction (0.25, 0.50, 1.00) on scaffold split."""
    rows = []
    for ds in ("bbbp", "bace"):
        for mdl in ("classical_matched", "hybrid_angle"):
            for frac in (0.25, 0.50, 1.0):
                sub = [
                    r for r in runs
                    if r["dataset"] == ds and r["model"] == mdl and r["split"] == "scaffold"
                    and np.isclose(r["train_fraction"], frac)
                    and r.get("n_qubits", 4) == 4 and r.get("n_layers", 3) == 3
                ]
                if not sub:
                    continue
                auc_m, auc_s = _mean_std([r["test"]["roc_auc"] for r in sub])
                pr_m, pr_s = _mean_std([r["test"]["pr_auc"] for r in sub])
                rows.append({
                    "Dataset": ds.upper(),
                    "Model": mdl,
                    "Train_Fraction": frac,
                    "Seeds": len(sub),
                    "ROC_AUC_Mean": auc_m,
                    "ROC_AUC_Std": auc_s,
                    "PR_AUC_Mean": pr_m,
                    "PR_AUC_Std": pr_s,
                })
    return pd.DataFrame(rows)


def build_qubit_depth_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Ablation C: Qubit & layer sweep for hybrid_angle on BACE scaffold split."""
    rows = []
    for nq in (4, 6):
        for nl in (1, 2, 3):
            sub = [
                r for r in runs
                if r["dataset"] == "bace" and r["model"] == "hybrid_angle"
                and r["split"] == "scaffold" and np.isclose(r["train_fraction"], 1.0)
                and r.get("n_qubits", 4) == nq and r.get("n_layers", 3) == nl
            ]
            if not sub:
                continue
            cnot, depth = get_circuit_specs(nq, nl)
            auc_m, auc_s = _mean_std([r["test"]["roc_auc"] for r in sub])
            pr_m, pr_s = _mean_std([r["test"]["pr_auc"] for r in sub])
            rows.append({
                "n_qubits": nq,
                "n_layers": nl,
                "Circuit_Params": 3 * nq * nl,
                "CNOT_Count": cnot,
                "Circuit_Depth": depth,
                "Seeds": len(sub),
                "ROC_AUC_Mean": auc_m,
                "ROC_AUC_Std": auc_s,
                "PR_AUC_Mean": pr_m,
                "PR_AUC_Std": pr_s,
            })
    return pd.DataFrame(rows)


def plot_split_leakage(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, ds in zip(axes, ("BBBP", "BACE")):
        sub = df[df["Dataset"] == ds]
        models = ["classical_matched", "hybrid_angle"]
        x = np.arange(len(models))
        width = 0.34
        for idx, (sp, color) in enumerate([("scaffold", "#2b5c8f"), ("random", "#e07b39")]):
            means = []
            stds = []
            for m in models:
                row = sub[(sub["Model"] == m) & (sub["Split"] == sp)]
                means.append(row["ROC_AUC_Mean"].values[0] if len(row) else 0.0)
                stds.append(row["ROC_AUC_Std"].values[0] if len(row) else 0.0)
            ax.bar(
                x + (idx - 0.5) * width,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=f"{sp.capitalize()} Split",
                color=color,
                alpha=0.9,
            )
        ax.set_title(f"{ds} — Split Leakage Audit")
        ax.set_xticks(x)
        ax.set_xticklabels(["Classical Matched", "Hybrid QGNN (Angle)"])
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Test ROC-AUC")
        ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_low_data(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    palette = {"classical_matched": "#2b5c8f", "hybrid_angle": "#1b9e77"}
    labels = {"classical_matched": "Classical Matched (41 head params)", "hybrid_angle": "Hybrid QGNN Angle (41 head params)"}
    for ax, ds in zip(axes, ("BBBP", "BACE")):
        sub = df[df["Dataset"] == ds]
        for mdl in ("classical_matched", "hybrid_angle"):
            m_sub = sub[sub["Model"] == mdl].sort_values("Train_Fraction")
            if m_sub.empty:
                continue
            xs = m_sub["Train_Fraction"].values * 100
            ys = m_sub["ROC_AUC_Mean"].values
            es = m_sub["ROC_AUC_Std"].values
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, lw=2, color=palette[mdl], label=labels[mdl])
        ax.set_title(f"{ds} — Low-Data Regime (Scaffold Split)")
        ax.set_xlabel("Training Set Fraction (%)")
        ax.set_ylabel("Test ROC-AUC")
        ax.set_xticks([25, 50, 100])
        ax.legend(frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qubit_depth(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    palette = {4: "#2b5c8f", 6: "#d95f02"}
    for nq in (4, 6):
        sub = df[df["n_qubits"] == nq].sort_values("n_layers")
        if sub.empty:
            continue
        ax.errorbar(
            sub["n_layers"].values,
            sub["ROC_AUC_Mean"].values,
            yerr=sub["ROC_AUC_Std"].values,
            marker="s",
            capsize=4,
            lw=2,
            color=palette[nq],
            label=f"{nq} Qubits (CNOTs: {sub['CNOT_Count'].tolist()})",
        )
    ax.set_title("BACE — Hybrid QGNN Qubit & Depth Sweep (Scaffold Split)")
    ax.set_xlabel("Variational Layers (L)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_xticks([1, 2, 3])
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_latex_tables(
    main_df: pd.DataFrame,
    split_df: pd.DataFrame,
    low_df: pd.DataFrame,
    qd_df: pd.DataFrame,
    out_path: Path,
) -> None:
    sections = [
        ("% Main Benchmark Table\n", main_df),
        ("% Ablation A: Split Leakage\n", split_df),
        ("% Ablation B: Low-Data Regime\n", low_df),
        ("% Ablation C: Qubit & Depth Sweep\n", qd_df),
    ]
    with out_path.open("w", encoding="utf-8") as f:
        for header, df in sections:
            f.write(header)
            if not df.empty:
                f.write(df.to_latex(index=False, float_format="%.4f"))
            f.write("\n\n")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs()
    main_df = build_main_table(runs)
    split_df = build_split_leakage_table(runs)
    low_df = build_low_data_table(runs)
    qd_df = build_qubit_depth_table(runs)

    main_df.to_csv(METRICS_DIR / "summary_main.csv", index=False)
    split_df.to_csv(METRICS_DIR / "ablation_split_leakage.csv", index=False)
    low_df.to_csv(METRICS_DIR / "ablation_low_data.csv", index=False)
    qd_df.to_csv(METRICS_DIR / "ablation_qubit_depth.csv", index=False)
    export_latex_tables(main_df, split_df, low_df, qd_df, METRICS_DIR / "summary_tables.tex")

    plot_split_leakage(split_df, FIGURES_DIR / "ablation_split_leakage.png")
    plot_low_data(low_df, FIGURES_DIR / "ablation_low_data.png")
    plot_qubit_depth(qd_df, FIGURES_DIR / "ablation_qubit_depth.png")

    print("\n=== MAIN BENCHMARK SUMMARY (Scaffold Split) ===")
    if not main_df.empty:
        print(main_df.to_string(index=False))

    print("\n=== ABLATION A: RANDOM VS. SCAFFOLD SPLIT LEAKAGE ===")
    if not split_df.empty:
        print(split_df.to_string(index=False))

    print("\n=== ABLATION B: LOW-DATA REGIME (Scaffold Split) ===")
    if not low_df.empty:
        print(low_df.to_string(index=False))

    print("\n=== ABLATION C: QUBIT & DEPTH SWEEP (BACE Scaffold Split) ===")
    if not qd_df.empty:
        print(qd_df.to_string(index=False))

    print("\n=== STATISTICAL SIGNIFICANCE (hybrid_angle vs. classical_matched, ROC-AUC) ===")
    for ds in ("bbbp", "bace"):
        for sp, frac in (("scaffold", 1.0), ("random", 1.0), ("scaffold", 0.50), ("scaffold", 0.25)):
            res = compute_paired_tests(runs, dataset=ds, split=sp, train_fraction=frac)
            print(
                f"{ds.upper():<4} | {sp:<8} | frac={frac:.2f} | n={res['n_seeds']} | "
                f"diff={res.get('mean_diff', float('nan')):+.4f} | "
                f"t_pval={res['t_pval']:.4f} | wilcoxon_pval={res['w_pval']:.4f}"
            )


if __name__ == "__main__":
    main()
