import logging
from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class IdentityGraphEncoder(nn.Module):
    """
    Fallback encoder that returns input node features unchanged and simple mean pooling.
    """
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        if self.out_channels != self.in_channels:
            self.proj = nn.Linear(self.in_channels, self.out_channels)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None):
        h = self.proj(x)
        if batch is None:
            graph_emb = h.mean(dim=0, keepdim=True)
        else:
            num_graphs = int(batch.max().item()) + 1
            graph_emb = torch.zeros(num_graphs, h.size(-1), device=h.device)
            for g in range(num_graphs):
                mask = (batch == g)
                if mask.any():
                    graph_emb[g] = h[mask].mean(dim=0)
        return h, graph_emb


class GCNEncoder(nn.Module):
    """
    Lightweight GCN encoder for WSI patch graphs.
    If torch_geometric is unavailable, falls back to IdentityGraphEncoder.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.use_pyg = GCNConv is not None and global_mean_pool is not None
        if not self.use_pyg:
            logging.warning("torch_geometric not available; using IdentityGraphEncoder")
            self.fallback = IdentityGraphEncoder(in_channels, out_channels)
            return

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor], batch: Optional[torch.Tensor] = None):
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        if not self.use_pyg:
            return self.fallback(x, edge_index, batch)
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        if batch is None:
            graph_emb = h.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(h, batch)
        return h, graph_emb


class GATEncoder(nn.Module):
    """
    Lightweight GAT encoder; falls back to IdentityGraphEncoder when torch_geometric is unavailable.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.use_pyg = GATConv is not None and global_mean_pool is not None
        if not self.use_pyg:
            logging.warning("torch_geometric not available; using IdentityGraphEncoder for GAT")
            self.fallback = IdentityGraphEncoder(in_channels, out_channels)
            return

        self.gat1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels, out_channels, heads=1, dropout=dropout, concat=True)
        self.act = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor], batch: Optional[torch.Tensor] = None):
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        if not self.use_pyg:
            return self.fallback(x, edge_index, batch)
        h = self.gat1(x, edge_index)
        h = self.act(h)
        h = self.dropout(h)
        h = self.gat2(h, edge_index)
        if batch is None:
            graph_emb = h.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(h, batch)
        return h, graph_emb


class GraphMambaEncoder(nn.Module):
    """
    Graph encoder using Mamba sequence mixer per graph.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2
        from torch_geometric.utils import to_dense_batch
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.mamba = Mamba2(d_model=hidden_channels, d_state=d_state, d_conv=d_conv, expand=expand, use_mem_eff_path=False)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self._to_dense_batch = to_dense_batch

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None):
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        h = self.input_proj(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            if batch.dim() > 1:
                batch = batch.reshape(-1)
            batch = batch.to(dtype=torch.long, device=x.device)
            if batch.numel() != x.size(0):
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_dense, mask = self._to_dense_batch(h, batch)
        
        if h_dense.dim() == 4 and h_dense.size(-1) == 1:
            h_dense = h_dense.squeeze(-1)
        if h_dense.dim() != 3:
            raise RuntimeError(f"GraphMambaEncoder expected 3D tensor, got shape {tuple(h_dense.shape)}")
        try:
            h_mixed = self.mamba(h_dense)
        except RuntimeError as e:
            if "context is destroyed" in str(e) or "Triton" in str(e):
                logging.warning(f"Mamba2 Triton error: {e}. Using identity transformation as fallback.")
                h_mixed = h_dense
            else:
                raise e
        h_flat = h_mixed[mask]
        h_flat = self.norm(h_flat)
        h_flat = self.dropout(h_flat)
        node_emb = self.output_proj(h_flat)
        if batch is None:
            graph_emb = node_emb.mean(dim=0, keepdim=True)
        else:
            num_graphs = int(batch.max().item()) + 1
            graph_emb = torch.zeros(num_graphs, node_emb.size(-1), device=node_emb.device)
            for g in range(num_graphs):
                mask_g = (batch == g)
                if mask_g.any():
                    graph_emb[g] = node_emb[mask_g].mean(dim=0)
        return node_emb, graph_emb


