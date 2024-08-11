import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import LossInfo, get_cov_std_loss

class VICRegLoss(nn.Module):
    def __init__(self, reconstruction_weight=100, std_weight=50):
        super(VICRegLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.std_weight = std_weight

    def forward(self, in_repr, out_repr, proj) -> LossInfo:
        reconstruction_loss = F.mse_loss(out_repr, in_repr)
        cov_std_loss = get_cov_std_loss(proj)

        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            cov_std_loss.cov_loss +
            self.std_weight * cov_std_loss.std_loss
        )

        diagnostics_info = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'cov_loss': cov_std_loss.cov_loss.item(),
            'std_loss': cov_std_loss.std_loss.item(),
        }

        return LossInfo(total_loss, diagnostics_info)
