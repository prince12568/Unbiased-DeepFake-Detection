"""CLIP ViT-B/16 (frozen) + LoRA (Eq. 3-4 of the UDD paper) + FC head.

The forward pass is split into embed -> run_blocks -> head so the shuffling
branch (hooks into `embed`) and the mixing branch (hooks between `run_blocks`
calls) can reuse exactly the same weights without touching this file's logic:

    tokens = model.embed(x)                       # S-branch replaces this
    tokens = model.run_blocks(tokens, 0, l)       # M-branch: mix tokens here
    tokens = model.run_blocks(tokens, l, None)
    logits, feat = model.head(tokens)
"""
import math

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    """y = W x + b + scale * B A x   (dW = B A, rank r)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


class LoRAQKV(nn.Module):
    """Separate rank-r LoRA for each of W_q, W_k, W_v on timm's fused qkv Linear
    (output layout is [q | k | v] along the last dim)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        dim = base.in_features
        assert base.out_features == 3 * dim
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(3, r, dim))
        self.B = nn.Parameter(torch.zeros(3, dim, r))
        with torch.no_grad():
            for k in range(3):
                nn.init.kaiming_uniform_(self.A[k], a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        h = torch.einsum("...d,krd->...kr", x, self.A)
        delta = torch.einsum("...kr,kor->...ko", h, self.B) * self.scale
        return out + delta.flatten(-2)


def add_lora(vit, r: int, alpha: float):
    """LoRA on q,k,v + attention proj (MSA) and fc1,fc2 (MLP) of every block."""
    for blk in vit.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r, alpha)
        blk.attn.proj = LoRALinear(blk.attn.proj, r, alpha)
        blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, r, alpha)
        blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, r, alpha)


class UDDViT(nn.Module):
    def __init__(
        self,
        arch="vit_base_patch16_clip_224.openai",
        pretrained=True,
        weights_file=None,
        lora_rank=4,
        lora_alpha=None,
        num_classes=2,
        grad_ckpt=False,
    ):
        super().__init__()
        kw = {}
        if pretrained and weights_file:
            kw["pretrained_cfg_overlay"] = dict(file=weights_file)
        self.vit = timm.create_model(
            arch, pretrained=pretrained, num_classes=0, drop_path_rate=0.0, **kw
        )
        for p in self.vit.parameters():
            p.requires_grad = False  # frozen backbone
        if lora_rank > 0:
            add_lora(self.vit, lora_rank, lora_alpha if lora_alpha else lora_rank)
        self.embed_dim = self.vit.embed_dim
        self.head_fc = nn.Linear(self.embed_dim, num_classes)
        self.grad_ckpt = grad_ckpt
        self.num_blocks = len(self.vit.blocks)

    # ---- pieces (hook points for S / M branches) ----
    def embed(self, x):
        t = self.vit.patch_embed(x)
        t = self.vit._pos_embed(t)  # adds CLS (index 0) + position embeddings
        t = self.vit.patch_drop(t)
        return self.vit.norm_pre(t)  # CLIP's ln_pre

    def run_blocks(self, t, start=0, end=None):
        for blk in self.vit.blocks[start:end]:
            if self.grad_ckpt and self.training:
                t = checkpoint(blk, t, use_reentrant=False)
            else:
                t = blk(t)
        return t

    def head(self, t):
        feat = self.vit.norm(t)[:, 0]  # [CLS] after final LN (t_cls)
        return self.head_fc(feat), feat

    def forward(self, x):
        return self.head(self.run_blocks(self.embed(x)))

    # ---- utils ----
    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def load_trainable(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        bad = [k for k in unexpected]
        assert not bad, f"unexpected keys in checkpoint: {bad[:5]}"
        return self

    def param_summary(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        return tr, tot
