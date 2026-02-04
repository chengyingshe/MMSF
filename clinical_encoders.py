import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch import Tensor
import torch.nn.functional as F


class ClinicalEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, n_aux_classes: List[int]):
        super().__init__()
        self.n_features = len(n_aux_classes)
        self.n_aux_classes = n_aux_classes
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        hidden_dim = hidden_dim
        for aux in n_aux_classes:
            self.encoder.append(nn.Sequential(
                nn.Linear(aux, hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim, bias=True)
            ))
            self.decoder.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, aux, bias=False)
            ))

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        # inputs: [B, sum(n_aux_classes)]
        pred_all: Optional[Tensor] = None
        hidden_all: Optional[Tensor] = None
        start = 0
        for i, aux in enumerate(self.n_aux_classes):
            end = start + aux
            x = inputs[:, start:end]
            h = self.encoder[i](x)
            pred = self.decoder[i](h)
            if pred_all is None:
                pred_all = pred
                hidden_all = h.unsqueeze(1)
            else:
                pred_all = torch.cat([pred_all, pred], dim=1)
                hidden_all = torch.cat([hidden_all, h.unsqueeze(1)], dim=1)
            start = end
        assert start == inputs.size(1), "ClinicalEmbedding: input feature size mismatch with n_aux_classes"
        emb_all = hidden_all.mean(dim=1)
        return emb_all, pred_all

def build_clinical_embedding_loss_fn(dataset):
    aux_dims = getattr(dataset, 'clinical_aux_dims', [])
    types = getattr(dataset, 'clinical_feature_types', [])
    def _loss(pred, target):
        loss = 0.0
        start = 0
        for aux_dim, ftype in zip(aux_dims, types):
            end = start + aux_dim
            p = pred[:, start:end]
            y = target[:, start:end]
            if ftype == 'num' or aux_dim == 1:
                loss = loss + F.mse_loss(p, y)
            else:
                y_idx = torch.argmax(y, dim=1)
                loss = loss + F.cross_entropy(p, y_idx, reduction='mean')
            start = end
        return loss
    return _loss
