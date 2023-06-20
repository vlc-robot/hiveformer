from typing import Optional, Tuple, Literal, Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride_size,
        apply_norm=True,
        apply_activation=True,
    ):
        super().__init__()

        padding_size = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else (kernel_size[0] // 2, kernel_size[1] // 2)
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride_size,
            padding_size,
            padding_mode="replicate",
        )

        if apply_norm:
            self.norm = nn.GroupNorm(1, out_channels, affine=True)

        if apply_activation:
            self.activation = nn.LeakyReLU(0.02)

    def forward(
        self, ft: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.conv(ft)

        if hasattr(self, "norm"):
            out = self.norm(out)

        if hasattr(self, "activation"):
            out = self.activation(out)

        return out


def dense_layer(in_channels, out_channels, apply_activation=True):
    layer: List[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer


def normalise_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


class ActionLoss(object):
    def decompose_actions(self, actions):
        pos = actions[..., :3]
        rot = actions[..., 3:7]
        open = actions[..., 7]
        return pos, rot, open

    def compute_loss(self, preds, targets, masks=None) -> Dict[str, torch.Tensor]:
        pred_pos, pred_rot, pred_open = self.decompose_actions(preds)
        tgt_pos, tgt_rot, tgt_open = self.decompose_actions(targets)

        losses = {}

        # Automatically matching the closest quaternions (symmetrical solution).
        tgt_rot_ = -tgt_rot.clone()
        
        if masks is None:
            losses['pos'] = F.mse_loss(pred_pos, tgt_pos)
    
            rot_loss = F.mse_loss(pred_rot, tgt_rot)
            rot_loss_ = F.mse_loss(pred_rot, tgt_rot_)
            select_mask = (rot_loss < rot_loss_).float()
            losses['rot'] = (select_mask * rot_loss + (1 - select_mask) * rot_loss_)
            losses['rot'] = rot_loss

            losses['open'] = F.binary_cross_entropy_with_logits(pred_open, tgt_open)
        else:
            div_sum = torch.sum(masks)

            losses['pos'] = torch.sum(F.mse_loss(
                pred_pos, tgt_pos, reduction='none') * masks.unsqueeze(-1)) / div_sum / 3
            
            rot_loss = torch.sum(F.mse_loss(
                pred_rot, tgt_rot, reduction='none') * masks.unsqueeze(-1))
            rot_loss_ = torch.sum(F.mse_loss(
                pred_rot, tgt_rot_, reduction='none') * masks.unsqueeze(-1))
            select_mask = (rot_loss < rot_loss_).float()
            losses['rot'] = (select_mask * rot_loss + (1 - select_mask) * rot_loss_) / div_sum / 4

            losses['open'] = torch.sum(F.binary_cross_entropy_with_logits(
                pred_open, tgt_open, reduction='none') * masks) / div_sum

        losses['total'] = losses['pos'] + losses['rot'] + losses['open']

        return losses
