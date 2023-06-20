from typing import Tuple
import numpy as np

import einops
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn

from models.network_utils import normalise_quat
from models.plain_unet import PlainUNet


class TransformerUNet(PlainUNet):
    def __init__(
        self, num_trans_layers: int = 1, nhead: int = 8,
        txt_attn_type: str = 'self', num_cams: int = 3,
        latent_im_size: Tuple = (8, 8),
        quat_input: str = 'add',  **kwargs
    ):
        self.num_trans_layers = num_trans_layers
        self.nhead = nhead
        self.txt_attn_type = txt_attn_type
        self.num_cams = num_cams
        self.latent_im_size = latent_im_size
        self.quat_input = quat_input

        super().__init__(**kwargs)

        if self.txt_attn_type == 'self':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.nhead,
                dim_feedforward=self.hidden_size*4,
                dropout=0.1, activation='gelu',
                layer_norm_eps=1e-12, norm_first=False,
                batch_first=True,
            )
            self.self_attention = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_trans_layers
            )

        elif self.txt_attn_type == 'cross':
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=self.nhead,
                dim_feedforward=self.hidden_size*4,
                dropout=0.1, activation='gelu',
                layer_norm_eps=1e-12, norm_first=False,
                batch_first=True,
            )
            self.cross_attention = nn.TransformerDecoder(
                decoder_layer, num_layers=self.num_trans_layers
            )

        self.visual_embedding = nn.Linear(self.enc_size, self.hidden_size)
        self.cam_embedding = nn.Embedding(self.num_cams, self.hidden_size)
        self.pix_embedding = nn.Embedding(
            np.prod(self.latent_im_size), self.hidden_size
        )

    @property
    def quat_hidden_size(self):
        if self.quat_input == 'add':
            return self.hidden_size * 3  # 3 cameras
        elif self.quat_input == 'concat':
            return self.hidden_size * 3 * 2
        else:
            raise NotImplementedError(
                'unsupport quat_input %s' % self.quat_input)

    def forward(self, batch, compute_loss=False):
        '''Inputs:
        - rgb_obs, pc_obs: (B, T, N, C, H, W) B: batch_size, N: #cameras
        - task_ids: (B, )
        - step_ids: (B, T)
        - step_masks: (B, T)
        - txt_masks: (B, L_txt)
        '''
        batch = self.prepare_batch(batch)
        device = batch['rgbs'].device

        rgb_obs = batch['rgbs']
        pc_obs = batch['pcds']
        taskvar_ids = batch['taskvar_ids']
        step_masks = batch['step_masks']
        step_ids = batch['step_ids']

        batch_size, nsteps, num_cams, _, im_height, im_width = rgb_obs.size()

        # (B, L, C)
        if 'instr_embeds' in batch:
            instr_embeds = self.task_embedding(batch['instr_embeds'])
            txt_masks = batch['txt_masks']
        else:
            instr_embeds = self.task_embedding(taskvar_ids).unsqueeze(1)
            txt_masks = torch.ones(batch_size, 1).long().to(device)

        rgb_fts = einops.rearrange(rgb_obs, "b t n c h w -> (b t n) c h w")
        rgb_fts = self.rgb_preprocess(rgb_fts)
        x = self.to_feat(rgb_fts)

        # encoding features
        enc_fts = []
        for l in self.feature_encoder:
            x = l(x)
            enc_fts.append(x)
        # x: (b t n) c h w
        x = einops.rearrange(
            x, '(b t n) c h w -> b t n (h w) c',
            b=batch_size, t=nsteps, n=num_cams
        )
        x = self.visual_embedding(x)

        enc_fts[-1] = einops.rearrange(x, "b t n (h w) c -> (b t n) c h w", h=self.latent_im_size[0], w=self.latent_im_size[1])

        step_embeds = self.step_embedding(batch['step_ids'])  # (B, T, C)
        cam_embeds = self.cam_embedding(
            torch.arange(num_cams).long().to(device)
        )   # (N, C)
        pix_embeds = self.pix_embedding(
            torch.arange(np.prod(self.latent_im_size)).long().to(device)
        )   # (H * W, C)
        x = x + einops.rearrange(step_embeds, 'b t c -> b t 1 1 c') + \
            einops.rearrange(cam_embeds, 'n c -> 1 1 n 1 c') + \
            einops.rearrange(pix_embeds, 'l c -> 1 1 1 l c')
        x = einops.rearrange(x, 'b t n l c -> b (t n l) c')

        # transformer: text / history encoding
        num_vtokens_per_step = num_cams * np.prod(self.latent_im_size)
        num_ttokens = instr_embeds.size(1)
        num_vtokens = num_vtokens_per_step * nsteps
        num_tokens = num_ttokens + num_vtokens

        if self.txt_attn_type == 'self':
            inputs = torch.cat(
                [instr_embeds, x], dim=1
            )

            ext_step_masks = einops.repeat(
                step_masks, 'b t -> b t k', k=num_vtokens_per_step
            ).flatten(start_dim=1)
            src_masks = torch.cat(
                [txt_masks, ext_step_masks], dim=1
            )   # (b, l+t*n*h*w)

            causal_masks = torch.zeros(num_tokens, num_tokens).bool()
            causal_masks[:num_ttokens, :num_ttokens] = True
            for t in range(nsteps):
                s = num_ttokens + num_vtokens_per_step * t
                e = s + num_vtokens_per_step
                causal_masks[s:e, :e] = True
            causal_masks = causal_masks.to(device)

            outputs = self.self_attention(
                inputs,
                mask=causal_masks.logical_not(),
                src_key_padding_mask=src_masks.logical_not()
            )
            outputs = outputs[:, num_ttokens:]

        elif self.txt_attn_type == 'cross':
            src_masks = einops.repeat(
                step_masks, 'b t -> b t k', k=num_vtokens_per_step
            ).flatten(start_dim=1)

            causal_masks = torch.zeros(num_vtokens, num_vtokens).bool()
            for t in range(nsteps):
                s = num_vtokens_per_step * t
                e = s + num_vtokens_per_step
                causal_masks[s:e, :e] = True
            causal_masks = causal_masks.to(device)

            outputs = self.cross_attention(
                x, instr_embeds,
                tgt_mask=causal_masks.logical_not(),
                tgt_key_padding_mask=src_masks.logical_not(),
                memory_key_padding_mask=txt_masks.logical_not(),
            )

        outputs = einops.rearrange(
            outputs, 'b (t n h w) c -> (b t n) c h w',
            t=nsteps, n=self.num_cams,
            h=self.latent_im_size[0], w=self.latent_im_size[1]
        )

        # decoding features
        enc_fts.reverse()

        if self.unet:
            xtr = outputs
            for i, l in enumerate(self.trans_decoder):
                xtr = l(torch.cat([xtr, enc_fts[i]], dim=1))

            # predict the translation of the gripper
            xt_heatmap = self.maps_to_coord(xtr)
            xt_heatmap = einops.rearrange(
                xt_heatmap, '(b t n) c h w -> b t (n c h w)', t=nsteps, n=num_cams, c=1
            )
            xt_heatmap = torch.softmax(xt_heatmap / 0.1, dim=-1)
            xt_heatmap = einops.rearrange(
                xt_heatmap, 'b t (n c h w) -> b t n c h w',
                t=nsteps, n=num_cams, c=1, h=im_height, w=im_width
            )
            xt = einops.reduce(pc_obs * xt_heatmap,
                               'b t n c h w -> b t c', 'sum')

        else:
            xt = 0

        # predict the (translation_offset, rotation and openness) of the gripper
        if self.quat_input == 'add':
            xg = outputs + enc_fts[0]
        elif self.quat_input == 'concat':
            xg = torch.cat([outputs, enc_fts[0]], dim=1)
        xg = einops.rearrange(
            xg, '(b t n) c h w -> (b t) (n c) h w', t=nsteps, n=num_cams
        )
        xg = self.quat_decoder(xg)
        xg = einops.rearrange(xg, '(b t) c -> b t c', t=nsteps)
        xt_offset = xg[..., :3]
        xr = normalise_quat(xg[..., 3:7])
        xo = xg[..., 7].unsqueeze(-1)

        actions = torch.cat([xt + xt_offset, xr, xo], dim=-1)

        if compute_loss:
            losses = self.loss_fn.compute_loss(
                actions, batch['actions'], masks=batch['step_masks']
            )
            return losses, actions

        return actions


if __name__ == '__main__':
    model = TransformerUNet(
        hidden_size=16, num_layers=4,
        num_tasks=None, max_steps=20,
        gripper_channel=False, unet=False,
        use_instr_embed='all', instr_embed_size=512,
        num_trans_layers=1, nhead=8,
        txt_attn_type='self', num_cams=3,
        latent_im_size=(8, 8)
    )

    b, t, n, h, w = 2, 4, 3, 128, 128
    batch = {
        'rgbs': torch.rand(b, t, n, 3, h, w),
        'pcds': torch.rand(b, t, n, 3, h, w),
        'taskvar_ids': torch.zeros(b).long(),
        'step_ids': einops.repeat(torch.arange(t).long(), 't -> b t', b=b),
        'step_masks': torch.ones(b, t).bool(),
        'instr_embeds': torch.rand(b, 5, 512),
        'txt_masks': torch.ones(b, 5).bool(),
    }

    actions = model(batch)
    print(actions.size())
    print(actions)
