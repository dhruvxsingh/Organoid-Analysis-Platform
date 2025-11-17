import torch
import torch.nn as nn
from dataclasses import dataclass
from dilipredict.nets import beitv2
from einops import rearrange, repeat


@dataclass
class STViTConfig:
    dim: int
    num_classes: int
    pool: str
    img_encoder_conf: beitv2.BeitV2Config


class STViT(nn.Module):

    def __init__(self, conf: STViTConfig) -> None:
        super().__init__()
        self.conf = conf
        self.seq_len = (conf.img_encoder_conf.img_size // conf.img_encoder_conf.patch_size) ** 2

        # image encoder
        self.image_transformer = beitv2.BeitV2(self.conf.img_encoder_conf)

        # spatial encoder
        self.space_blocks = nn.ModuleList([
            beitv2.ViTEncoderBlock(
                dim=self.conf.dim, num_heads=12, init_values=0.1
            ) for _ in range(2)
        ])
        self.space_token = nn.Parameter(torch.randn(1, 1, self.conf.dim))

        # temporal encoder
        self.temporal_lstm = nn.LSTM(
            input_size=self.conf.dim, hidden_size=self.conf.dim, bidirectional=True, batch_first=True
        )

        self.mlp = beitv2.Mlp(self.seq_len, self.seq_len, 1)
        self.fc = nn.Linear(self.conf.dim * 2, self.conf.num_classes)

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(torch.load(ckpt_files[0], map_location=torch.device('cpu')))

    def load_pretrained_ckpt(self, beitv2_ckpt: str, freeze_beitv2: bool = False) -> None:
        self.image_transformer.load_ckpt(beitv2_ckpt)
        if freeze_beitv2:
            self.image_transformer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x.shape: (b c t s h w)
            b: batch_size
            c: image channel
            t: image time num
            s: image spatial num
            h: image height
            w: image width
        '''
        b, c, t, s, h, w = x.shape
        x = rearrange(x, 'b c t s h w -> (b t s) c h w')
        x = self.image_transformer.forward_feature(x)
        if self.conf.pool == 'cls':
            x = rearrange(x[:, 0], '(bt s) ... -> bt s ...', bt=b * t)
        else:
            x = x[:, 1:]
            x = rearrange(x, 'b n d -> b d n')
            x = self.mlp(x)
            x = rearrange(x, '(bt s) d n -> bt (s n) d', bt=b * t)
        cls_space_tokens = repeat(
            self.space_token, '() n d -> bt n d', bt=b * t)
        x = torch.cat((cls_space_tokens, x), dim=1)
        for blk in self.space_blocks:
            x = blk(x)

        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        packed_output, (hidden, cell) = self.temporal_lstm(x)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden)
