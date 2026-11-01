"""mix_branch.py -- Mixing Branch (paper: Token-Mixing Branch).

Inserted in the MIDDLE of the network (paper Table 4: mid stage is best, 'before first block' is worst):
after a randomly chosen block k in [mix_layer_min, mix_layer_max] (1-indexed, default 5..8 of 12),
30% of the target image's patch tokens are replaced by the partner's tokens (partner = another sample
of the SAME label in the batch). The target's CLS token is never touched.
"""
import torch


def _same_label_pairs(labels):
    B = labels.size(0)
    pair_idx = torch.arange(B, device=labels.device)
    for lbl in labels.unique():
        mask = (labels == lbl).nonzero(as_tuple=True)[0]
        if mask.size(0) < 2:
            continue                       # no same-label partner in this (micro-)batch: sample is left unmixed
        shuffled = mask[torch.randperm(mask.size(0), device=labels.device)]
        for k in range(mask.size(0)):
            if shuffled[k] == mask[k]:
                swap = (k + 1) % mask.size(0)
                shuffled[k], shuffled[swap] = shuffled[swap].clone(), shuffled[k].clone()
        pair_idx[mask] = shuffled
    return pair_idx


def token_swap(tokens, pair_idx, ratio=0.3):
    """Paper mechanism (two INDEPENDENT random subsets, not a same-slot swap):
      1) drop_pos:   gammaN target-token slots are dropped (I_D in the paper)
      2) src_pos:    gammaN source tokens are independently selected (T^src'_l)
      3) merge:      the dropped target slots are filled with those selected source
                      tokens -> T^m_l = T^tgt'_l union T^src'_l
    drop_pos and src_pos are sampled independently per sample, so a filled slot's
    partner token does NOT have to come from that same spatial position."""
    B, N, D = tokens.shape
    N_patch = N - 1
    n_swap = max(1, int(ratio * N_patch))
    mixed = tokens.clone()
    partner = tokens[pair_idx]
    for i in range(B):
        drop_pos = torch.randperm(N_patch, device=tokens.device)[:n_swap] + 1   # +1: skip CLS
        src_pos = torch.randperm(N_patch, device=tokens.device)[:n_swap] + 1    # independent sample
        mixed[i, drop_pos] = partner[i, src_pos]
    return mixed


def mix_forward(model, x, labels, ratio=0.3, layer_range=(5, 8)):
    """Returns (logits_mix, feat_mix, pair_idx). The original branch is NOT recomputed here."""
    lo, hi = layer_range
    hi = min(hi, model.num_blocks - 1)
    k = int(torch.randint(lo, hi + 1, (1,)).item())       # swap after block k
    tokens = model.run_blocks(model.embed(x), 0, k)
    pair_idx = _same_label_pairs(labels)
    tokens = token_swap(tokens, pair_idx, ratio)
    logits_mix, feat_mix = model.head(model.run_blocks(tokens, k, None))
    return logits_mix, feat_mix, pair_idx


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from model import UDDViT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = UDDViT(pretrained=False).to(device).eval()
    x = torch.randn(8, 3, 224, 224, device=device)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)
    with torch.no_grad():
        _, fo = m(x)
        lm, fm, pidx = mix_forward(m, x, y)
    print("logits_mix:", tuple(lm.shape), "| feat diff orig vs mix:", (fo - fm).norm(dim=1).mean().item())
    assert (fo - fm).norm(dim=1).mean().item() > 0.01, "FAIL: mixing had no effect"
    print("PASS")
