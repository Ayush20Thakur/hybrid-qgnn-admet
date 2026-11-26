"""Modular GNN encoder for molecular graphs (GIN / GATv2) with bounded latent projection."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, global_max_pool, global_mean_pool


class MolecularGNNEncoder(nn.Module):
    """3-layer GINEConv / GATv2Conv encoder with dual pooling and [-pi, pi] bottleneck."""

    def __init__(
        self,
        node_dim: int = 45,
        edge_dim: int = 12,
        hidden_dim: int = 64,
        latent_dim: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        arch: Literal["gin", "gat"] = "gin",
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.arch = arch.lower()

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            if self.arch == "gin":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
            elif self.arch == "gat":
                self.convs.append(
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // 4,
                        heads=4,
                        edge_dim=hidden_dim,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(f"Unsupported GNN arch {arch!r}. Use 'gin' or 'gat'.")
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Dual pooling (mean + max) -> 2 * hidden_dim -> latent_dim
        self.bottleneck = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of molecular graphs into [-pi, pi]^latent_dim."""
        h = F.relu(self.node_proj(x))
        e = F.relu(self.edge_proj(edge_attr))

        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h_out = conv(h, edge_index, e)
            h_out = F.relu(bn(h_out))
            h_out = F.dropout(h_out, p=self.dropout, training=self.training)
            h = h_in + h_out  # Residual connection

        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        latent = torch.tanh(self.bottleneck(pooled)) * math.pi
        return latent
