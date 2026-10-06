"""Historical direct-gated variant, different from PADL Algorithm 1.

Coefficients have shape [K, N]. Reconstruction uses elementwise gates;
the historical regularizer sums column maxima rather than Algorithm 1 row maxima.
"""

import torch

def _init_R_beta(K, n_samples, alpha=1, beta=3.7, device='cuda'):
    """使用Beta分布初始化R到logit空间"""
    beta_dist = torch.distributions.Beta(alpha, beta)
    R_beta = beta_dist.sample((K, n_samples)).to(device)
    # 转换到logit空间
    R_raw = torch.log(R_beta / (1 - R_beta + 1e-8))
    return R_raw
def _clamp_R(R_raw):
    """Sigmoid 归一化到 [0,1]"""
    return torch.sigmoid(R_raw)

def apply_delta_gate(R, delta: float, tau: float = 20.0, hard: bool = False):
    """
    用 delta 作为 value-bar 来决定激活（替代 topk）。

    Args:
        R:        [K, N] in [0,1]
        delta:    阈值（value bar），建议 0.2~0.8 之间按分布调
        tau:      soft gate 温度，越大越接近硬阈值
        hard:     True 时用硬阈值；False 时用 soft sigmoid gate

    Returns:
        R_gated:  [K, N]
        gate:     [K, N]（soft 或 hard 的 gate）
    """
    if hard:
        gate = (R >= delta).to(R.dtype)
    else:
        gate = torch.sigmoid(tau * (R - delta))
    return R * gate, gate


def reconstruction_loss_delta(X, R_raw, D, delta=None, tau: float = 20.0):
    """
    用 delta gate 的重建损失（替代 reconstruction_loss_sparse 的 topk 方案）
    """
    R = _clamp_R(R_raw)  # [K, N]

    if delta is not None:
        R_eff, _ = apply_delta_gate(R, delta=delta, tau=tau, hard=False)  # soft, 可微
    else:
        R_eff = R

    pred = D @ R_eff
    return torch.norm(X - pred, p='fro') ** 2


def delta_usage_penalty(R_raw, delta: float, tau: float = 20.0, target_sparsity: float = None):
    """
    可选：对“激活比例”做一个轻量约束，避免 delta 选不好导致全开或全关。
    - target_sparsity: 希望激活比例（0~1），例如 0.05 表示约 5% 系数被激活
    """
    R = _clamp_R(R_raw)
    _, gate = apply_delta_gate(R, delta=delta, tau=tau, hard=False)  # gate in (0,1)
    act_ratio = gate.mean()  # 近似激活比例

    if target_sparsity is None:
        # 不指定目标时，返回 0（即不约束）
        return act_ratio * 0.0

    # L2 拉到目标激活比例
    return (act_ratio - target_sparsity) ** 2


def R_loss_delta(X, R_raw, D, delta=None, tau: float = 20.0,
                 beta_weight=0.1,
                 include_beta_reg: bool = True,
                 target_sparsity: float = None,
                 sparsity_weight: float = 0.0):
    """
    R 的总损失：把重建项换成 delta gate 版本，其余沿用你原来的正则。
    """
    mu_value = 1
    lambda_l1 = 1.33
    lambda_linf = 1.01

    recon = mu_value * reconstruction_loss_delta(X, R_raw, D, delta=delta, tau=tau)

    R = _clamp_R(R_raw)
    beta_reg = -beta_weight * torch.mean((0.5-1)*torch.log(R+1e-8) + torch.log(1-R+1e-8))

    # 可选：激活比例约束（不强制开）
    sp_pen = delta_usage_penalty(R_raw, delta=delta, tau=tau, target_sparsity=target_sparsity)
    sp_pen = sparsity_weight * sp_pen

    # 保持你原有 l1 / dictionary_size 正则
    loss = recon + lambda_l1 * l1_loss(R_raw) + lambda_linf * dictionary_size(R_raw)
    if include_beta_reg:
        loss = loss + beta_reg
    return loss + sp_pen


def l1_loss(R_raw):
    R = _clamp_R(R_raw)
    return torch.norm(R, p=1) / R.numel()

def dictionary_size(R_raw):
    R = _clamp_R(R_raw)
    return torch.sum(torch.max(R, dim=0).values)

class DirectGatedDictionary:
    def __init__(self, d_size, K, num_iters, device, delta=None, tau=20.0,
                 target_sparsity: float = None, sparsity_weight: float = 0.0,
                 beta: float = 3.5, beta_weight: float = 0.1,
                 include_beta_reg: bool = True, predict_iters: int = 100,
                 r_steps: int = 20, d_steps: int = 10,
                 lr_r: float = 5e-3, lr_d: float = 5e-4):
        """
        delta: 用 value-bar 控制激活（替代 sparsity=topk）
        tau:   soft gate 温度
        target_sparsity / sparsity_weight: 可选的激活比例约束
        """
        self.d_size = d_size
        self.device = device
        self.iters  = num_iters
        self.K = K

        self.delta = delta
        self.tau = tau

        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
        self.beta = beta
        self.beta_weight = beta_weight
        self.include_beta_reg = include_beta_reg
        self.predict_iters = predict_iters

        self.D = torch.rand(self.d_size, self.K, device=device)
        self.D.requires_grad_(True)

        self.k_r = r_steps
        self.k_d = d_steps
        self.lr_r = lr_r
        self.lr_d = lr_d

    def _apply_delta(self, R):
        """
        推理阶段：硬阈值 delta gate（真正稀疏）
        """
        if self.delta is None:
            return R
        R_hard, _ = apply_delta_gate(R, delta=self.delta, tau=self.tau, hard=True)
        return R_hard

    def predict(self, X):
        x_mean = X.mean(dim=0, keepdim=True)
        x_std = X.std(dim=0, keepdim=True) + 1e-8
        X_norm = (X - x_mean) / x_std

        _, n_samples = X_norm.shape

        # 你的 beta init（保持一致）
        R = _init_R_beta(self.K, n_samples, beta=self.beta, device=self.device)
        R.requires_grad_(True)

        optimizer_R = torch.optim.Adam([R], lr=self.lr_r)

        for _ in range(self.predict_iters):
            optimizer_R.zero_grad()
            loss_r = R_loss_delta(
                X_norm, R, self.D,
                delta=self.delta, tau=self.tau,
                beta_weight=self.beta_weight,
                include_beta_reg=self.include_beta_reg,
                target_sparsity=self.target_sparsity,
                sparsity_weight=self.sparsity_weight
            )
            loss_r.backward()
            optimizer_R.step()

        R_final = _clamp_R(R)
        R_final = self._apply_delta(R_final)  # hard delta gate

        X_recon_norm = self.D @ R_final
        X_recon = X_recon_norm * x_std + x_mean
        return self.D.detach(), R_final.detach(), X_recon.detach()

    def fit(self, X):
        x_mean = X.mean(dim=0, keepdim=True)
        x_std = X.std(dim=0, keepdim=True) + 1e-8
        X_norm = (X - x_mean) / x_std

        _, n_samples = X_norm.shape
        R = _init_R_beta(self.K, n_samples, beta=self.beta, device=self.device)
        R.requires_grad_(True)

        optimizer_R = torch.optim.Adam([R], lr=self.lr_r)
        optimizer_D = torch.optim.Adam([self.D], lr=self.lr_d)

        for i in range(self.iters):
            # update R
            for _ in range(self.k_r):
                optimizer_R.zero_grad()
                loss_r = R_loss_delta(
                    X_norm, R, self.D,
                    delta=self.delta, tau=self.tau,
                    beta_weight=self.beta_weight,
                    include_beta_reg=self.include_beta_reg,
                    target_sparsity=self.target_sparsity,
                    sparsity_weight=self.sparsity_weight
                )
                loss_r.backward()
                optimizer_R.step()

            # update D: 用 soft gate 的重建项（可微）
            for _ in range(self.k_d):
                optimizer_D.zero_grad()
                loss_d = 10.0 * reconstruction_loss_delta(
                    X_norm, R, self.D, delta=self.delta, tau=self.tau
                )
                loss_d.backward()
                optimizer_D.step()

            with torch.no_grad():
                self.D.data.clamp_(0, 1)
                self.D /= torch.norm(self.D, dim=0, keepdim=True) + 1e-8

        return self.D.detach()
