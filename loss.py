import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_Similarity


class Cross_inscl_loss(nn.Module):
    def __init__(self, tau=0.5, eps=1e-8, alpha=0.6, use_symKL=True,
                 use_conf_weight=True, lambda_conf=3.0):
        """
        alpha: weight for JS (if use_symKL True, total loss = alpha*JS + (1-alpha)*symKL)
        use_symKL: whether to include symmetric KL in interpolation
        use_conf_weight: whether to apply confidence weighting
        """
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.alpha = alpha
        self.use_symKL = use_symKL
        self.use_conf_weight = use_conf_weight
        self.lambda_conf = lambda_conf

    def forward(self, h0, h1):
        h0 = nn.functional.normalize(h0, dim=1)
        h1 = nn.functional.normalize(h1, dim=1)
        cos12 = get_Similarity(h0, h1)
        cos11 = get_Similarity(h0, h0)
        cos21 = get_Similarity(h1, h0)
        cos22 = get_Similarity(h1, h1)

        sim12 = (cos12 / self.tau).exp()
        sim11 = (cos11 / self.tau).exp()
        sim21 = (cos21 / self.tau).exp()
        sim22 = (cos22 / self.tau).exp()

        pos1 = sim12.diag()
        pos2 = sim21.diag()

        p1 = pos1 / (sim12 + sim11).sum(1)
        p2 = pos2 / (sim21 + sim22).sum(1)

        # clamp then normalize across samples to make proper distributions
        p1 = p1.clamp(min=self.eps)
        p2 = p2.clamp(min=self.eps)
        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()

        # mixture distribution M
        m = 0.5 * (p1 + p2)
        m = m.clamp(min=self.eps)

        if self.use_conf_weight:
            # confidence as average prob per sample
            conf = 0.5 * (p1 + p2)
            w = torch.sigmoid(-self.lambda_conf * (1.0 - conf))
            w = w / w.mean()
        else:
            w = torch.ones_like(p1)

        kl_p1_m = (w * p1 * (torch.log(p1 / m))).sum()
        kl_p2_m = (w * p2 * (torch.log(p2 / m))).sum()
        L_js = 0.5 * (kl_p1_m + kl_p2_m)

        if self.use_symKL:
            kl_p1_p2 = (w * p1 * (torch.log(p1 / p2))).sum()
            kl_p2_p1 = (w * p2 * (torch.log(p2 / p1))).sum()
            symKL = 0.5 * (kl_p1_p2 + kl_p2_p1)
            loss = self.alpha * L_js + (1.0 - self.alpha) * symKL
        else:
            loss = L_js

        return loss





class Noise_robust_loss(nn.Module):
    def __init__(self, alpha=0.5 , eps=1e-12):
        """
        alpha: 控制在 MAE (alpha=1) 和 InfoNCE (alpha->0) 之间平滑过渡
        eps: 避免 log(0) 或 0^alpha
        """
        super(Noise_robust_loss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, h0, h1):
        # 1. 归一化向量
        h0, h1 = nn.functional.normalize(h0, dim=1), nn.functional.normalize(h1, dim=1)
        tao = 1.0

        # 2. 相似度矩阵
        cos = get_Similarity(h0, h1)   # shape: [N,N]
        sim = (cos / tao).exp()

        # 3. 正样本概率 Q
        pos = sim.diag()
        Q = pos / sim.sum(1)
        Q = Q.clamp(min=self.eps, max=1.0)

        # 4. Tsallis/GCE 桥接损失
        if abs(self.alpha - 1.0) < 1e-12:
            loss = 2.0 * (1.0 - Q)            # alpha=1 时 -> MAE
        else:
            loss = 2.0 * (1.0 - Q.pow(self.alpha)) / self.alpha  # 0<alpha<1 -> 介于两者之间

        robust_loss = loss.mean()
        return robust_loss

