"""shuf_branch.py -- Shuffling Branch (paper: Token-Shuffling Branch, Eq. 2).

t_i = pos'_i + e_pi(i)
  * e_pi(i): patch-content tokens, shuffled in 2x2 blocks (blocks permuted, patches inside a block stay together)
  * pos'_i : position embeddings cropped to a random rectangle (area >= ~30% of the grid, random aspect
             ratio and location) and bilinearly resized back to the 14x14 grid
CLS token and its position embedding are untouched. Everything after the embedding is identical to
model.embed(): patch_drop -> CLIP ln_pre (norm_pre) -> blocks.
"""
import torch
import torch.nn.functional as F


def _block_shuffle_2x2(patch_content, grid_h=14, grid_w=14):
    """(B, N, D) -> (B, N, D): independent random permutation of the 2x2 blocks per sample."""
    B, N, D = patch_content.shape
    bh, bw = grid_h // 2, grid_w // 2
    n_blocks = bh * bw
    x = patch_content.view(B, bh, 2, bw, 2, D).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, n_blocks, 4, D)
    idx = torch.stack([torch.randperm(n_blocks, device=patch_content.device) for _ in range(B)])
    x = x.gather(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, D))
    x = x.view(B, bh, bw, 2, 2, D).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, N, D)


def random_interp_pos(patch_pos, B, g=14, area=(60.0, 196.0), ratio=(0.75, 4.0 / 3.0)):
    """Random crop + bilinear resize of the patch position embeddings (paper Table 5: aspect ratio
    (3/4, 4/3), area S in [60, 196] cells, random location). patch_pos: (1, g*g, D) -> (B, g*g, D)."""
    dev, D = patch_pos.device, patch_pos.shape[-1]
    S = torch.empty(B, device=dev).uniform_(*area)
    r = torch.empty(B, device=dev).uniform_(*ratio)
    w = torch.sqrt(S * r).clamp(max=g)           # crop width / height in grid cells
    h = torch.sqrt(S / r).clamp(max=g)
    x0 = torch.rand(B, device=dev) * (g - w)     # left / top, always inside the grid
    y0 = torch.rand(B, device=dev) * (g - h)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w / g
    theta[:, 0, 2] = (2 * x0 + w) / g - 1
    theta[:, 1, 1] = h / g
    theta[:, 1, 2] = (2 * y0 + h) / g - 1
    grid = F.affine_grid(theta, (B, D, g, g), align_corners=False)
    pos = patch_pos.float().reshape(1, g, g, D).permute(0, 3, 1, 2).expand(B, -1, -1, -1)
    out = F.grid_sample(pos, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.permute(0, 2, 3, 1).reshape(B, g * g, D).to(patch_pos.dtype)


def shuf_forward(model, x, pos_crop=True):
    """Shuffling-branch forward. Returns (logits, feat) like model(x)."""
    vit = model.vit
    B = x.size(0)
    patches = vit.patch_embed(x)                                  # (B, N, D), content only
    g = int(round(patches.shape[1] ** 0.5))
    patches = _block_shuffle_2x2(patches, g, g)
    assert vit.pos_embed.shape[1] == patches.shape[1] + 1, "expected pos_embed with a CLS entry"
    patch_pos = vit.pos_embed[:, 1:]
    if pos_crop:
        patch_pos = random_interp_pos(patch_pos, B, g)
    pos = torch.cat([vit.pos_embed[:, :1].expand(B, -1, -1), patch_pos.expand(B, -1, -1)], dim=1)
    tokens = torch.cat([vit.cls_token.expand(B, -1, -1), patches], dim=1) + pos
    tokens = vit.norm_pre(vit.patch_drop(vit.pos_drop(tokens)))   # same tail as model.embed()
    return model.head(model.run_blocks(tokens))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from model import UDDViT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = UDDViT(pretrained=False).to(device).eval()
    x = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        lb, fb = m(x)
        ls, fs = shuf_forward(m, x)
    print("logits_shuf:", tuple(ls.shape), "| differs from normal path:", (lb - ls).abs().max().item() > 1e-4)
    assert (lb - ls).abs().max().item() > 1e-4, "FAIL: shuffling had no effect"
    print("PASS")
