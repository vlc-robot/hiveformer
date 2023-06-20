from typing import Optional, Tuple, Literal, Union, List, Dict

import numpy as np
from einops.layers.torch import Rearrange
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network_utils import (
    ConvLayer, dense_layer, normalise_quat,
    ActionLoss
)


class PlainUNet(nn.Module):
    def __init__(
        self, hidden_size: int = 16, num_layers: int = 4,
        num_tasks: int = None, max_steps: int = 20,
        gripper_channel: bool = False, unet: bool = True,
        use_instr_embed: str = 'none', instr_embed_size: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.max_steps = max_steps
        self.gripper_channel = gripper_channel
        self.unet = unet
        self.use_instr_embed = use_instr_embed
        self.instr_embed_size = instr_embed_size

        if self.use_instr_embed == 'none':
            assert self.num_tasks is not None
            self.task_embedding = nn.Embedding(
                self.num_tasks, self.hidden_size)
        else:
            assert self.instr_embed_size is not None
            self.task_embedding = nn.Linear(
                self.instr_embed_size, self.hidden_size)

        self.step_embedding = nn.Embedding(self.max_steps, self.hidden_size)

        # in_channels: RGB or RGB + gripper pose heatmap image
        self.in_channels = 4 if self.gripper_channel == "attn" else 3

        # Input RGB Preprocess (SiameseNet for each camera)
        self.rgb_preprocess = ConvLayer(
            self.in_channels, self.hidden_size // 2,
            kernel_size=(3, 3),
            stride_size=(1, 1),
            apply_norm=False,
        )
        self.to_feat = ConvLayer(
            self.hidden_size // 2, self.hidden_size,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
        )

        # Encoder-Decoder Network, maps to pixel location with spatial argmax
        self.feature_encoder = nn.ModuleList()
        for i in range(self.num_layers):
            self.feature_encoder.append(
                ConvLayer(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=(3, 3),
                    stride_size=(2, 2)
                )
            )

        self.enc_size = self.hidden_size

        if self.unet:
            self.trans_decoder = nn.ModuleList()
            for i in range(self.num_layers):
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels=self.hidden_size * 2,
                                out_channels=self.hidden_size,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )

        quat_hidden_size = self.quat_hidden_size
        self.quat_decoder = nn.Sequential(
            ConvLayer(
                in_channels=quat_hidden_size, out_channels=quat_hidden_size,
                kernel_size=(3, 3), stride_size=(2, 2),
            ),
            ConvLayer(
                in_channels=quat_hidden_size, out_channels=quat_hidden_size,
                kernel_size=(3, 3), stride_size=(2, 2)
            ),
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c h w -> b (c h w)"),
            *dense_layer(quat_hidden_size, quat_hidden_size),
            *dense_layer(quat_hidden_size, 3 + 4 + 1, apply_activation=False),
        )

        self.maps_to_coord = ConvLayer(
            in_channels=self.hidden_size,
            out_channels=1,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
            apply_activation=False,
        )

        self.loss_fn = ActionLoss()

    @property
    def quat_hidden_size(self):
        return self.hidden_size * 4  # 3 cameras + task/step embedding

    @property
    def num_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            nweights += np.prod(v.size())
            nparams += 1
        return nweights, nparams

    @property
    def num_trainable_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            if v.requires_grad:
                nweights += np.prod(v.size())
                nparams += 1
        return nweights, nparams

    def prepare_batch(self, batch):
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch

    def forward(self, batch, compute_loss=False):
        '''Inputs:
        - rgb_obs, pc_obs: (B, N, C, H, W) B: batch_size, N: #cameras
        - task_ids, step_ids: (B, )
        '''
        batch = self.prepare_batch(batch)

        rgb_obs = batch['rgbs']
        pc_obs = batch['pcds']
        taskvar_ids = batch['taskvar_ids']
        step_ids = batch['step_ids']
        instr_embeds = batch.get('instr_embeds', None)

        batch_size, n_cameras, _, im_height, im_width = rgb_obs.size()

        rgb_fts = einops.rearrange(rgb_obs, "b n c h w -> (b n) c h w")
        rgb_fts = self.rgb_preprocess(rgb_fts)
        x = self.to_feat(rgb_fts)

        # encoding features
        enc_fts = []
        for l in self.feature_encoder:
            x = l(x)
            enc_fts.append(x)

        # concat the rgb fts with task/step embeds
        if self.use_instr_embed == 'none':
            task_embeds = self.task_embedding(taskvar_ids)
        else:
            assert instr_embeds.size(1) == 1
            task_embeds = self.task_embedding(instr_embeds[:, 0])
        step_embeds = self.step_embedding(step_ids)
        task_step_embeds = task_embeds + step_embeds

        # decoding features
        enc_fts.reverse()

        if self.unet:
            ext_task_step_embeds = einops.repeat(
                task_step_embeds, 'b c -> (b n) c h w',
                n=n_cameras, h=enc_fts[0].shape[-2], w=enc_fts[0].shape[-1]
            )
            x = torch.cat([enc_fts[0], ext_task_step_embeds], dim=1)

            for i, l in enumerate(self.trans_decoder):
                if i == 0:
                    xtr = l(x)
                else:
                    xtr = l(torch.cat([xtr, enc_fts[i]], dim=1))

            # predict the translation of the gripper
            xt_heatmap = self.maps_to_coord(xtr)
            xt_heatmap = einops.rearrange(
                xt_heatmap, '(b n) c h w -> b (n c h w)', n=n_cameras, c=1
            )
            xt_heatmap = torch.softmax(xt_heatmap / 0.1, dim=1)
            xt_heatmap = einops.rearrange(
                xt_heatmap, 'b (n c h w) -> b n c h w',
                n=n_cameras, c=1, h=im_height, w=im_width
            )
            xt = einops.reduce(pc_obs * xt_heatmap, 'b n c h w -> b c', 'sum')

        else:
            xt = 0

        # predict the (translation_offset, rotation and openness) of the gripper
        xg = einops.rearrange(
            enc_fts[0], '(b n) c h w -> b (n c) h w', n=n_cameras
        )
        ext_task_step_embeds = einops.repeat(
            task_step_embeds, 'b c -> b c h w',
            h=xg.size(2), w=xg.size(3)
        )
        xg = torch.cat([xg, ext_task_step_embeds], dim=1)
        xg = self.quat_decoder(xg)
        xt_offset = xg[..., :3]
        xr = normalise_quat(xg[..., 3:7])
        xo = xg[..., 7].unsqueeze(-1)

        actions = torch.cat([xt + xt_offset, xr, xo], dim=-1)

        if compute_loss:
            losses = self.loss_fn.compute_loss(actions, batch['actions'])
            return losses, actions

        return actions


if __name__ == '__main__':
    model = PlainUNet(
        hidden_size=16, num_layers=4,
        num_tasks=1, max_steps=20,
        gripper_channel=False, unet=True,
        use_instr_embed='avg', instr_embed_size=512,
    )
    print(next(model.parameters()).device)

    b, n, h, w = 2, 3, 128, 128
    batch = {
        'rgbs': torch.rand(b, n, 3, h, w),
        'pcds': torch.rand(b, n, 3, h, w),
        'taskvar_ids': torch.zeros(b).long(),
        'step_ids': torch.randint(0, 10, (b, )).long(),
        'instr_embeds': torch.rand(b, 1, 512),
    }

    actions = model(batch)
    print(actions.size())
    print(actions)
