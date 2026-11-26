"""Seeded training and evaluation runner for Classical and Hybrid QGNN models."""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.data.featurizer import get_feature_dims, load_moleculenet_dataset
from src.data.scaffold_split import bemis_murcko_scaffold_split, load_split_indices, random_split
from src.models.classical_head import ClassicalGNNFull, ClassicalGNNParamMatched, count_parameters
from src.models.gnn_encoder import MolecularGNNEncoder
from src.models.hybrid_qgnn import HybridQGNN

warnings.filterwarnings("ignore", message="The usage of `scatter\\(reduce='max'\\)`")
warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated")

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
ALL_MODELS = ["classical_full", "classical_matched", "hybrid_angle", "hybrid_reuploading"]


def set_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(train_graphs: list[Any], device: torch.device) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight = n_neg / max(n_pos, 1) on training split."""
    labels = [float(g.y.item()) for g in train_graphs]
    n_pos = sum(1 for y in labels if y == 1.0)
    n_neg = len(labels) - n_pos
    weight = n_neg / max(n_pos, 1)
    return torch.tensor([weight], dtype=torch.float32, device=device)


def build_model(
    model_name: str,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int = 64,
    n_qubits: int = 4,
    n_layers: int = 3,
    dropout: float = 0.2,
    arch: str = "gin",
    quantum_device: str = "lightning.qubit",
) -> nn.Module:
    """Instantiate the requested baseline or hybrid model."""
    encoder = MolecularGNNEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        latent_dim=n_qubits,
        num_layers=3,
        dropout=dropout,
        arch=arch,
    )
    if model_name == "classical_full":
        return ClassicalGNNFull(encoder=encoder, mlp_hidden=32, num_tasks=1, dropout=dropout)
    if model_name == "classical_matched":
        return ClassicalGNNParamMatched(encoder=encoder, n_layers=n_layers, num_tasks=1)
    if model_name == "hybrid_angle":
        return HybridQGNN(
            encoder=encoder,
            n_layers=n_layers,
            num_tasks=1,
            encoding_type="angle",
            quantum_device=quantum_device,
        )
    if model_name == "hybrid_reuploading":
        return HybridQGNN(
            encoder=encoder,
            n_layers=n_layers,
            num_tasks=1,
            encoding_type="reuploading",
            quantum_device=quantum_device,
        )
    raise ValueError(f"Unsupported model {model_name!r}.")


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate loss, ROC-AUC, PR-AUC, F1, and Accuracy on a PyG DataLoader."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y)
        bs = batch.y.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        probs = torch.sigmoid(logits).view(-1).cpu().numpy()
        targets = batch.y.view(-1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets)

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    return {
        "loss": total_loss / max(n_samples, 1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "accuracy": acc,
    }


def train_single_seed(
    dataset_name: str,
    model_name: str,
    seed: int,
    split_type: str = "scaffold",
    train_fraction: float = 1.0,
    hidden_dim: int = 64,
    n_qubits: int = 4,
    n_layers: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 20,
    dropout: float = 0.2,
    metrics_dir: str | Path = "results/metrics",
    checkpoints_dir: str | Path = "results/checkpoints",
    quantum_device: str = "lightning.qubit",
    quiet: bool = False,
) -> dict[str, Any]:
    """Train one model for one seed, save best checkpoint and JSON metrics."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_moleculenet_dataset(dataset_name)
    node_dim, edge_dim = get_feature_dims()

    if split_type == "scaffold":
        split_dir = Path("data/processed") / f"{dataset_name.lower()}_scaffold"
        if (split_dir / "train_idx.npy").exists():
            train_idx, val_idx, test_idx = load_split_indices(dataset_name, split_type="scaffold")
        else:
            train_idx, val_idx, test_idx = bemis_murcko_scaffold_split(graphs)
    elif split_type == "random":
        train_idx, val_idx, test_idx = random_split(len(graphs), seed=seed)
    else:
        raise ValueError(f"Unknown split_type {split_type!r}")

    if train_fraction < 1.0:
        rng = np.random.default_rng(seed)
        k = max(1, int(round(len(train_idx) * train_fraction)))
        train_idx = rng.choice(train_idx, size=k, replace=False)

    train_graphs = [graphs[int(i)] for i in train_idx]
    val_graphs = [graphs[int(i)] for i in val_idx]
    test_graphs = [graphs[int(i)] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    model = build_model(
        model_name=model_name,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        dropout=dropout,
        quantum_device=quantum_device,
    ).to(device)
    param_counts = count_parameters(model, verbose=not quiet)

    pos_weight = compute_pos_weight(train_graphs, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bs = batch.y.size(0)
            train_loss += loss.item() * bs
            n_train += bs

        scheduler.step()
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(n_train, 1),
                "val_loss": val_metrics["loss"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    elapsed = time.perf_counter() - start_time
    ql_tag = "" if (n_qubits == 4 and n_layers == 3) else f"_q{n_qubits}_l{n_layers}"
    if best_state is not None:
        # Load state dicts cleanly (preserving quantum_layer CPU placement for HybridQGNN)
        model.load_state_dict(best_state)
        model.to(device)

        ckpt_path = Path(checkpoints_dir) / f"{dataset_name.lower()}_{model_name}{ql_tag}_seed{seed}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "dataset": dataset_name.lower(),
                "model": model_name,
                "seed": seed,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
            },
            ckpt_path,
        )

    test_metrics = evaluate_loader(model, test_loader, criterion, device)
    val_final = evaluate_loader(model, val_loader, criterion, device)

    result = {
        "dataset": dataset_name.lower(),
        "model": model_name,
        "split": split_type,
        "train_fraction": train_fraction,
        "seed": seed,
        "device": str(device),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "params": param_counts,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "wall_time_sec": round(elapsed, 2),
        "val": val_final,
        "test": test_metrics,
        "history": history,
    }

    exp_name = f"{dataset_name.lower()}_{model_name}_{split_type}_f{int(train_fraction*100)}{ql_tag}_seed{seed}"
    out_path = Path(metrics_dir) / f"{exp_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if not quiet:
        print(
            f"[{dataset_name.upper()} | {model_name} | seed={seed}] "
            f"best_ep={best_epoch:02d} | val_auc={best_val_auc:.4f} | "
            f"test_auc={test_metrics['roc_auc']:.4f} | pr_auc={test_metrics['pr_auc']:.4f} | "
            f"f1={test_metrics['f1']:.4f} ({elapsed:.1f}s)"
        )
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Mean +/- Std summary across seeds for a group of runs."""
    aucs = [r["test"]["roc_auc"] for r in results]
    prs = [r["test"]["pr_auc"] for r in results]
    f1s = [r["test"]["f1"] for r in results]
    return {
        "dataset": results[0]["dataset"].upper(),
        "model": results[0]["model"],
        "params_total": results[0]["params"]["total"],
        "params_head": results[0]["params"]["head"],
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "pr_auc_mean": float(np.mean(prs)),
        "pr_auc_std": float(np.std(prs, ddof=1)) if len(prs) > 1 else 0.0,
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
    }


def print_comparison_table(summaries: list[dict[str, Any]], title: str) -> None:
    """Print a clean Mean +/- Std comparison table."""
    print("\n" + "=" * 98)
    print(title)
    print("-" * 98)
    print(
        f"{'Dataset':<8} | {'Model':<20} | {'Head/Total Params':>17} | "
        f"{'Test ROC-AUC':>15} | {'Test PR-AUC':>15} | {'Test F1':>15}"
    )
    print("-" * 98)
    for s in summaries:
        p_str = f"{s['params_head']:,} / {s['params_total']:,}"
        auc_str = f"{s['roc_auc_mean']:.4f} ± {s['roc_auc_std']:.4f}"
        pr_str = f"{s['pr_auc_mean']:.4f} ± {s['pr_auc_std']:.4f}"
        f1_str = f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
        print(
            f"{s['dataset']:<8} | {s['model']:<20} | {p_str:>17} | "
            f"{auc_str:>15} | {pr_str:>15} | {f1_str:>15}"
        )
    print("=" * 98)


def load_saved_summaries(
    datasets: list[str],
    models: list[str],
    split_type: str = "scaffold",
    train_fraction: float = 1.0,
    seeds: list[int] = DEFAULT_SEEDS,
    metrics_dir: str | Path = "results/metrics",
) -> list[dict[str, Any]]:
    """Load saved per-seed JSON metrics and summarize them."""
    summaries: list[dict[str, Any]] = []
    for ds in datasets:
        for mdl in models:
            runs = []
            for seed in seeds:
                p = Path(metrics_dir) / f"{ds}_{mdl}_{split_type}_f{int(train_fraction*100)}_seed{seed}.json"
                if p.exists():
                    with p.open("r", encoding="utf-8") as f:
                        runs.append(json.load(f))
            if runs:
                summaries.append(summarize_results(runs))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Classical & Hybrid QGNN models.")
    parser.add_argument("--datasets", nargs="+", default=["bbbp", "bace"], choices=["bbbp", "bace"])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--quantum_device", default="lightning.qubit")
    parser.add_argument("--compare_all", action="store_true", help="Include saved classical baselines in table.")
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for ds in args.datasets:
        for mdl in args.models:
            runs = []
            for seed in args.seeds:
                runs.append(
                    train_single_seed(
                        dataset_name=ds,
                        model_name=mdl,
                        seed=seed,
                        split_type=args.split,
                        train_fraction=args.train_fraction,
                        n_qubits=args.n_qubits,
                        n_layers=args.n_layers,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        max_epochs=args.epochs,
                        patience=args.patience,
                        quantum_device=args.quantum_device,
                    )
                )
            summaries.append(summarize_results(runs))

    if args.compare_all:
        all_summaries = load_saved_summaries(
            datasets=args.datasets,
            models=ALL_MODELS,
            split_type=args.split,
            train_fraction=args.train_fraction,
            seeds=args.seeds,
        )
        print_comparison_table(
            all_summaries,
            f"Classical vs. Hybrid QGNN Comparison ({args.split} split, {len(args.seeds)} seeds)",
        )
    else:
        print_comparison_table(
            summaries,
            f"Training Summary ({args.split} split, {len(args.seeds)} seeds)",
        )


if __name__ == "__main__":
    main()
