import torch
from segquant.utils.psquare import PSquare


class AnomalyChannelDetector:
    def __init__(self, k: int, alpha=0.5, device=None, p=50):
        self.k = k
        self.alpha = alpha
        self.max = torch.zeros(k, dtype=torch.float32, device=device)
        # self.mdns = PSquare(k, p=p, device=device)
        self.mus = torch.zeros(k, dtype=torch.float32, device=device)
        self.m2s = torch.zeros(k, dtype=torch.float32, device=device)
        self.n = 0
        self.weight_norm_2 = None
        self.weight_norm_2_squared = None

    def update(self, X: torch.Tensor, W: torch.Tensor):
        if self.weight_norm_2 is None and self.weight_norm_2_squared is None:
            W = W.t()  # (in, out)
            self.weight_norm_2 = torch.norm(W, dim=1, p=2)
            self.weight_norm_2_squared = (W**2).sum(dim=1)

        this_batch = X.shape[0]
        X_abs = X.to(dtype=torch.float32, device=self.mus.device).abs() # (b, k)

        ## Welford's algorithm -- batch version
        mu_batch = torch.mean(X_abs, dim=0)  # (k,)
        M2_batch = ((X_abs - mu_batch) ** 2).sum(dim=0)  # (k,)
        delta = mu_batch - self.mus
        n_old = self.n
        self.n += this_batch
        self.mus += delta * this_batch / self.n
        self.m2s += M2_batch + delta**2 * n_old * this_batch / self.n

        # ## P² algorithm
        # for x_line in X_abs:
        #     self.mdns.update(x_line) # (k,)

        self.max = torch.maximum(self.max, X_abs.max(dim=0).values)

    def get_anomaly_scores(self):
        eps = 1e-8

        fishers = (
            self.m2s / (self.n - 1)
            if self.n > 1
            else torch.zeros_like(self.m2s) * self.weight_norm_2_squared
        ) # （k,)

        fishers_norm = fishers / (fishers.sum() + eps)
        # mdns = self.mdns.p_estimate() # (k,)
        mdns = self.mus
        tails = self.max / (mdns + eps) * self.weight_norm_2  # (k,)
        tails_norm = tails / (tails.sum() + eps)

        scores = self.alpha * tails_norm + (1 - self.alpha) * fishers_norm
        return scores
