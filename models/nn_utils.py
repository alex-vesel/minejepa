from typing import NamedTuple
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn.functional as F


class CovStdLoss(NamedTuple):
    cov_loss: torch.Tensor
    std_loss: torch.Tensor


@dataclass
class LossInfo:
    total_loss: torch.Tensor
    diagnostics_info: dict


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_cov_std_loss(x: torch.Tensor) -> CovStdLoss:
    batch_size = x.shape[0]
    num_features = x.shape[-1]

    x = x - x.mean(dim=0)

    std = torch.sqrt(x.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std))

    cov = (x.T @ x) / (batch_size - 1)
    cov_loss = off_diagonal(cov).pow_(2).sum().div(num_features)

    return CovStdLoss(cov_loss, std_loss)


def concat_loss_infos(loss_infos):
    new_loss_info = LossInfo(
        total_loss=torch.stack([l.total_loss for l in loss_infos]).mean(),
        diagnostics_info={k: float(np.mean([l.diagnostics_info[k] for l in loss_infos])) for k in loss_infos[0].diagnostics_info},
    )
    return new_loss_info