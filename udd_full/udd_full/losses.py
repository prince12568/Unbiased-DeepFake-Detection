"""losses.py -- UDD losses (paper Eq. 5-8).

L_con   = L_c(g(t), g(t_s)) + L_c(g(t), g(t_m))            (SimCLR-style InfoNCE, g = 3-layer MLP, tau = 0.1)
L_align = JS(P_orig || P_shuf) + JS(P_orig || P_mix)       (Jensen-Shannon on the softmax outputs)
L_total = CE + lambda1 * L_con + lambda2 * L_align         (CE on all three branches)
Gradients flow through ALL branches, including the original one (the paper uses no stop-gradient).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """3-layer MLP g(.): D -> D -> D -> proj_dim."""
    def __init__(self, in_dim=768, hidden_dim=768, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim))

    def forward(self, x):
        return self.net(x)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def pair(self, za, zb):
        """Symmetric InfoNCE: positive = same image in the other view, negatives = other images in the batch."""
        sim = za @ zb.T / self.tau
        tgt = torch.arange(za.size(0), device=za.device)
        return 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt))

    def forward(self, z_orig, z_shuf, z_mix):
        return self.pair(z_orig, z_shuf) + self.pair(z_orig, z_mix)


def js_div(logits_p, logits_q):
    p = F.softmax(logits_p.float(), dim=1)
    q = F.softmax(logits_q.float(), dim=1)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + 1e-8) - torch.log(m + 1e-8))).sum(1).mean()
    kl_qm = (q * (torch.log(q + 1e-8) - torch.log(m + 1e-8))).sum(1).mean()
    return 0.5 * (kl_pm + kl_qm)


class UDDLoss(nn.Module):
    def __init__(self, embed_dim=768, proj_dim=128, head_fc=None, temperature=0.1,
                 lambda1=0.1, lambda2=0.1, class_weight=None):
        super().__init__()
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.register_buffer("class_weight", class_weight)
        self.projector = ProjectionHead(embed_dim, embed_dim, proj_dim)
        self.infonce = InfoNCELoss(temperature)
        # head_fc kept in the signature for compatibility with the notebook; it is no longer needed

    def forward(self, logits_base, logits_shuf, logits_mix, feat_orig, feat_shuf, feat_mix, labels):
        cw = self.class_weight
        ce_base = F.cross_entropy(logits_base.float(), labels, weight=cw)
        ce_shuf = F.cross_entropy(logits_shuf.float(), labels, weight=cw)
        ce_mix = F.cross_entropy(logits_mix.float(), labels, weight=cw)
        z_orig = F.normalize(self.projector(feat_orig), dim=1)
        z_shuf = F.normalize(self.projector(feat_shuf), dim=1)
        z_mix = F.normalize(self.projector(feat_mix), dim=1)
        l_con = self.infonce(z_orig, z_shuf, z_mix)
        l_js = js_div(logits_base, logits_shuf) + js_div(logits_base, logits_mix)
        total = ce_base + ce_shuf + ce_mix + self.lambda1 * l_con + self.lambda2 * l_js
        return {"loss": total, "ce_base": ce_base.detach(), "ce_shuf": ce_shuf.detach(),
                "ce_mix": ce_mix.detach(), "l_con": l_con.detach(), "l_js": l_js.detach()}


if __name__ == "__main__":
    B, D = 8, 768
    head_fc = nn.Linear(D, 2)
    crit = UDDLoss(embed_dim=D, head_fc=head_fc)
    lb, ls, lm = torch.randn(B, 2), torch.randn(B, 2), torch.randn(B, 2)
    fo, fs, fm = torch.randn(B, D), torch.randn(B, D), torch.randn(B, D)
    y = torch.randint(0, 2, (B,))
    out = crit(lb, ls, lm, fo, fs, fm, y)
    for k, v in out.items():
        print(f"  {k:10s}: {v.item():.4f}")
    assert abs(js_div(lb, lb).item()) < 1e-6
    print("PASS")
