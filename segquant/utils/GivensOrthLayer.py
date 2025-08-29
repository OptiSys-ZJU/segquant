import math
import torch
import torch.nn as nn


class GivensOrthLayer(nn.Module):
    def __init__(
        self,
        k,
        givens_num=None,
        sample_mode="rand",
        sample_func=None,
        init_vecs_mode="identity",
        generator=None,
        dtype=torch.float32,
        device=None,
        requires_grad=True,
        enable_autograd=False,
        enable_grad_buffer=False,
        enable_low_memory_grad=False,
    ):
        super(GivensOrthLayer, self).__init__()

        self.k = k  # problem dimension
        self.dof = k * (k - 1) // 2  # degrees of freedom

        if givens_num is None:
            self.givens_num = self.dof
        else:
            assert givens_num <= self.dof, f"givens_num should be <= {self.dof}"
            self.givens_num = givens_num

        assert sample_mode in [
            "rand",
            "ascending",
            "descending",
            "custom",
        ], "sample_mode should be one of ['rand', 'ascending', 'descending', 'custom']"

        all_pairs = GivensOrthLayer.generate_upper_triangular_pairs(k)
        if sample_mode == "custom":
            assert sample_func is not None
            indices = sample_func(k, self.givens_num)
        else:
            indices = GivensOrthLayer.sample_pairs_indices(
                k, self.givens_num, sample_mode=sample_mode, generator=generator
            )
        self.pairs = [all_pairs[i] for i in indices]
        self.device = device
        self.dtype = dtype
        self.enable_autograd = enable_autograd
        self.enable_grad_buffer = enable_grad_buffer
        self.enable_low_memory_grad = enable_low_memory_grad

        assert init_vecs_mode in ["identity", "random"], "Unknown init mode"
        self.vecs = nn.Parameter(
            GivensOrthLayer.init_vecs(
                self.givens_num,
                mode=init_vecs_mode,
                dtype=dtype,
                device=device,
                generator=generator,
            ),
            requires_grad=enable_autograd,
        )

        if requires_grad and self.enable_grad_buffer:
            # buffer for prefix and suffix products
            assert (
                not enable_low_memory_grad
            ), "enable_low_memory_grad is not compatible with enable_grad_buffer"
            self.register_buffer("cs", torch.empty((self.givens_num, 2), device=device, dtype=dtype))
            self.register_buffer("P", torch.eye(k, device=device, dtype=dtype).unsqueeze(0).repeat(self.givens_num,1,1))
            self.register_buffer("S", torch.eye(k, device=device, dtype=dtype).unsqueeze(0).repeat(self.givens_num,1,1))
            self.register_buffer("dG_dc", torch.tensor([[1.0,0.0],[0.0,1.0]], device=device, dtype=dtype))
            self.register_buffer("dG_ds", torch.tensor([[0.0,-1.0],[1.0,0.0]], device=device, dtype=dtype))
            GivensOrthLayer.cal_cs_inplace(self.cs, self.vecs)

    @staticmethod
    def init_vecs(n, mode="identity", dtype=torch.float32, device=None, generator=None):
        if mode == "identity":
            x = torch.ones(n, 1, device=device, dtype=dtype)
            y = torch.zeros(n, 1, device=device, dtype=dtype)
        elif mode == "random":
            theta = (
                2
                * math.pi
                * torch.rand(n, device=device, dtype=dtype, generator=generator)
            )
            x = torch.cos(theta).unsqueeze(1)
            y = torch.sin(theta).unsqueeze(1)
        else:
            raise ValueError("Unknown init mode")
        return torch.cat([x, y], dim=1)  # shape (k, 2)

    @staticmethod
    def generate_upper_triangular_pairs(k):
        pairs = []
        for i in range(k - 1):
            for j in range(i + 1, k):
                pairs.append((i, j))
        return pairs

    @staticmethod
    def sample_pairs_indices(k, m, sample_mode="rand", generator=None):
        dof = k * (k - 1) // 2
        if sample_mode == "rand":
            perm = torch.randperm(dof, generator=generator, device=device)
            return perm[:m].tolist()
        elif sample_mode == "ascending":
            return list(range(m))
        elif sample_mode == "descending":
            return list(range(dof - m, dof))
        else:
            raise ValueError(
                "sample_mode should be one of ['rand', 'ascending', 'descending']"
            )

    @staticmethod
    def cal_cs_inplace(cs, vecs):
        r = torch.hypot(vecs[:, 0], vecs[:, 1])
        cs[:, 0] = vecs[:, 0] / r
        cs[:, 1] = vecs[:, 1] / r
        return cs

    @staticmethod
    def build_givens_matrix(k, pair, cs, dtype=torch.float32, device=None):
        G = torch.eye(k, dtype=dtype, device=device)
        i, j = pair
        G[i, i] = cs[i, 0]
        G[j, j] = cs[j, 0]
        G[i, j] = -cs[i, 1]
        G[j, i] = cs[j, 1]
        return G

    @staticmethod
    def build_Q_slow(k, pairs, cs, dtype=torch.float32, device=None):
        Q = torch.eye(k, dtype=dtype, device=device)
        for idx, (p, q) in enumerate(pairs):
            ci, si = cs[idx, 0], cs[idx, 1]
            G = torch.eye(k, dtype=dtype, device=device)
            G[p, p] = ci
            G[q, q] = ci
            G[p, q] = -si
            G[q, p] = si
            Q = Q @ G

        return Q

    @staticmethod
    def build_Q(k, pairs, cs, dtype=torch.float32, device=None):
        Q = torch.eye(k, dtype=dtype, device=device)
        for idx, (p, q) in enumerate(pairs):
            ci, si = cs[idx, 0], cs[idx, 1]
            Qp, Qq = Q[:, p].clone(), Q[:, q]
            Q[:, p] = ci * Qp + si * Qq
            Q[:, q] = -si * Qp + ci * Qq

        return Q

    @staticmethod
    def grad_slow(
        k, pairs, vecs, cs, dtype=torch.float32, device=None, chain_grad=None
    ):
        m = len(pairs)
        dG_dc = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)
        dG_ds = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=device, dtype=dtype)
        g = torch.zeros((m, 2), device=device, dtype=dtype)  # gradient w.r.t. (x, y)
        for j in range(m):
            P = torch.eye(k, device=device, dtype=dtype)
            S = torch.eye(k, device=device, dtype=dtype)
            p, q = pairs[j]
            x, y = vecs[j]
            r = torch.hypot(x, y)
            for i in range(j):
                pi, qi = pairs[i]
                ci, si = cs[i, 0], cs[i, 1]
                Pp, Pq = P[:, pi].clone(), P[:, qi]
                P[:, pi] = ci * Pp + si * Pq
                P[:, qi] = -si * Pp + ci * Pq
            for i in range(m - 1 - j):
                pi, qi = pairs[m - 1 - i]
                ci, si = cs[m - 1 - i, 0], cs[m - 1 - i, 1]
                Sp, Sq = S[pi, :].clone(), S[qi, :]
                S[pi, :] = ci * Sp - si * Sq
                S[qi, :] = si * Sp + ci * Sq

            L = P[:, [p, q]]  # (k,2)
            R = S[[p, q], :]  # (2,k)

            # gradient w.r.t c and s
            if chain_grad is not None:
                g_c = torch.sum(chain_grad * (L @ dG_dc @ R))
                g_s = torch.sum(chain_grad * (L @ dG_ds @ R))
            else:
                g_c = torch.sum(L @ dG_dc @ R)
                g_s = torch.sum(L @ dG_ds @ R)

            # chain rule to x,y
            g[j, 0] = g_c * (y**2 / r**3) - g_s * (x * y / r**3)
            g[j, 1] = -g_c * (x * y / r**3) + g_s * (x**2 / r**3)
        return g

    @staticmethod
    def grad(k, pairs, vecs, cs, dtype=torch.float32, device=None, chain_grad=None):
        m = len(pairs)
        P = [torch.eye(k, device=device, dtype=dtype)]
        for j in range(m - 1):
            c, s = cs[j, 0], cs[j, 1]
            p, q = pairs[j]
            Pj = P[-1].clone()
            Pj[:, p] = c * P[-1][:, p] + s * P[-1][:, q]
            Pj[:, q] = -s * P[-1][:, p] + c * P[-1][:, q]
            P.append(Pj)

        S = [torch.eye(k, device=device, dtype=dtype)]
        for j in range(m - 1):
            c, s = cs[m - 1 - j, 0], cs[m - 1 - j, 1]
            p, q = pairs[m - 1 - j]
            Sj = S[-1].clone()
            Sj[p, :] = c * S[-1][p, :] - s * S[-1][q, :]
            Sj[q, :] = s * S[-1][p, :] + c * S[-1][q, :]
            S.append(Sj)

        g = torch.zeros((m, 2), device=device, dtype=dtype)  # gradient w.r.t. (x, y)
        dG_dc = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)
        dG_ds = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=device, dtype=dtype)
        for j in range(m):
            p, q = pairs[j]
            x, y = vecs[j]
            r = torch.hypot(x, y)

            L = P[j][:, [p, q]]  # (k,2)
            R = S[m - 1 - j][[p, q], :]  # (2,k)

            # gradient w.r.t c and s
            if chain_grad is not None:
                g_c = torch.sum(chain_grad * (L @ dG_dc @ R))
                g_s = torch.sum(chain_grad * (L @ dG_ds @ R))
            else:
                g_c = torch.sum(L @ dG_dc @ R)
                g_s = torch.sum(L @ dG_ds @ R)

            # chain rule to x,y
            g[j, 0] = g_c * (y**2 / r**3) - g_s * (x * y / r**3)
            g[j, 1] = -g_c * (x * y / r**3) + g_s * (x**2 / r**3)

        return g

    def _grad_buffer(self, pairs, vecs, cs, chain_grad=None):
        m = len(pairs)
        for j in range(m - 1):
            c, s = cs[j, 0], cs[j, 1]
            p, q = pairs[j]
            self.P[j + 1, :, :].copy_(self.P[j, :, :])
            self.P[j + 1, :, p] = c * self.P[j, :, p] + s * self.P[j, :, q]
            self.P[j + 1, :, q] = -s * self.P[j, :, p] + c * self.P[j, :, q]

        for j in range(m - 1):
            c, s = cs[m - 1 - j, 0], cs[m - 1 - j, 1]
            p, q = pairs[m - 1 - j]
            self.S[j + 1, :, :].copy_(self.S[j, :, :])
            self.S[j + 1, p, :] = c * self.S[j, p, :] - s * self.S[j, q, :]
            self.S[j + 1, q, :] = s * self.S[j, p, :] + c * self.S[j, q, :]

        g = torch.zeros(
            (m, 2), device=self.device, dtype=self.dtype
        )  # gradient w.r.t. (x, y)
        for j in range(m):
            p, q = pairs[j]
            x, y = vecs[j]
            r = torch.hypot(x, y)

            L = self.P[j][:, [p, q]]  # (k,2)
            R = self.S[m - 1 - j][[p, q], :]  # (2,k)

            # gradient w.r.t c and s
            if chain_grad is not None:
                g_c = torch.sum(chain_grad * (L @ self.dG_dc @ R))
                g_s = torch.sum(chain_grad * (L @ self.dG_ds @ R))
            else:
                g_c = torch.sum(L @ self.dG_dc @ R)
                g_s = torch.sum(L @ self.dG_ds @ R)

            # chain rule to x,y
            g[j, 0] = g_c * (y**2 / r**3) - g_s * (x * y / r**3)
            g[j, 1] = -g_c * (x * y / r**3) + g_s * (x**2 / r**3)

        return g

    def try_grad(self, chain_grad=None):
        k = self.k
        pairs = self.pairs
        vecs = self.vecs
        if self.enable_grad_buffer:
            cs = self.cs
        else:
            cs = torch.empty((self.givens_num, 2), device=self.device, dtype=self.dtype)
            GivensOrthLayer.cal_cs_inplace(cs, self.vecs)
        if not self.enable_grad_buffer:
            if self.enable_low_memory_grad:
                return GivensOrthLayer.grad_slow(
                    k,
                    pairs,
                    vecs,
                    cs,
                    dtype=self.dtype,
                    device=self.device,
                    chain_grad=chain_grad,
                )
            return GivensOrthLayer.grad(
                k,
                pairs,
                vecs,
                cs,
                dtype=self.dtype,
                device=self.device,
                chain_grad=chain_grad,
            )

        return self._grad_buffer(pairs, vecs, cs, chain_grad=chain_grad)

    def forward(self):
        if self.enable_grad_buffer:
            cs = self.cs
        else:
            cs = torch.empty((self.givens_num, 2), device=self.device, dtype=self.dtype)

        GivensOrthLayer.cal_cs_inplace(cs, self.vecs)
        if self.enable_autograd:
            Q = GivensOrthLayer.build_Q_slow(
                self.k, self.pairs, cs, dtype=self.dtype, device=self.device
            )
            return Q
        else:
            Q = GivensOrthLayer.build_Q(
                self.k, self.pairs, cs, dtype=self.dtype, device=self.device
            )
        return Q

if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    device = "cpu"
    m, k, n = 2, 32, 32
    givens = 10
    update_steps = 10
    sample_mode = "rand"
    init_mode = "random"

    grad = 'buffer'  # 'buffer', 'normal', 'slow'

    if grad == 'buffer':
        enable_grad_buffer = True
        enable_low_memory_grad = False
    elif grad == 'normal':
        enable_grad_buffer = False
        enable_low_memory_grad = False
    elif grad == 'slow':
        enable_grad_buffer = False
        enable_low_memory_grad = True
    else:
        raise ValueError("Unknown grad mode")

    print(f"Testing GivensManager with k={k}, givens={givens}, sample_mode={sample_mode}, init_mode={init_mode}, grad={grad}")

    X = torch.randn(m, k, device=device)
    W = torch.randn(k, n, device=device)
    epsX = 0.01 * torch.randn(m, k, device=device)
    epsW = 0.01 * torch.randn(k, n, device=device)

    def loss(manager, X, W, epsX, epsW):
        Q = manager.forward()
        term = X @ Q @ epsW + epsX @ Q.t() @ W + epsX @ epsW
        return torch.norm(term, p="fro") ** 2, Q

    @torch.no_grad()
    def manual_grad_Q(Q, X, W, epsX, epsW):
        XtX = X.t() @ X
        WWT = W @ W.t()
        epsXtX = epsX.t() @ epsX
        epsWDW = epsW @ epsW.t()

        grad = 2 * (XtX @ Q @ epsWDW + WWT @ Q @ epsXtX)
        grad += 2 * (X.t() @ epsX @ Q.t() @ W @ epsW.t() + W @ epsW.t() @ Q.t() @ X.t() @ epsX)
        grad += 2 * (X.t() @ epsX @ epsWDW + W @ epsW.t() @ epsXtX)
        return grad

    @torch.no_grad()
    def manual_grad(manager, X, W, epsX, epsW):
        Q = manager.forward()
        gQ = manual_grad_Q(Q, X, W, epsX, epsW)
        g = manager.try_grad(chain_grad=gQ)
        return g

    manual_manager = GivensOrthLayer(
        k,
        givens_num=givens,
        sample_mode=sample_mode,
        init_vecs_mode=init_mode,
        generator=torch.Generator().manual_seed(42),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
        enable_autograd=False,
        enable_grad_buffer=enable_grad_buffer,
        enable_low_memory_grad=enable_low_memory_grad,
    )

    auto_manager = GivensOrthLayer(
        k,
        givens_num=givens,
        sample_mode=sample_mode,
        init_vecs_mode=init_mode,
        generator=torch.Generator().manual_seed(42),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
        enable_autograd=True,
        enable_grad_buffer=False,
        enable_low_memory_grad=False,
    )

    opt_manual = torch.optim.SGD([manual_manager.vecs], lr=0.1)
    opt_auto = torch.optim.SGD([auto_manager.vecs], lr=0.1)

    print("Initial m-vecs:", manual_manager.vecs, manual_manager.pairs)
    print("Initial a-vecs:", auto_manager.vecs, auto_manager.pairs)

    manual_time = 0.0
    auto_time = 0.0

    for step in range(update_steps):
        # ---- manual ----
        L_manual, Q_manual = loss(manual_manager, X, W, epsX, epsW)
        start_manual = time.time()
        opt_manual.zero_grad()
        g_manual = manual_grad(manual_manager, X, W, epsX, epsW)
        with torch.no_grad():
            manual_manager.vecs.grad = g_manual
        opt_manual.step()
        manual_time += time.time() - start_manual
        g_manual = g_manual.clone()

        # ---- autograd ----
        start_auto = time.time()
        opt_auto.zero_grad()
        L_auto, Q_auto = loss(auto_manager, X, W, epsX, epsW)
        L_auto.backward()
        opt_auto.step()
        auto_time += time.time() - start_auto
        g_auto = auto_manager.vecs.grad.clone()

        # ---- diff ----
        loss_diff = torch.norm(L_manual.detach() - L_auto.detach()).item()
        grad_diff = torch.norm(g_manual - g_auto).item()
        vec_diff = torch.norm(
            manual_manager.vecs.detach() - auto_manager.vecs.detach()
        ).item()
        Q_diff = torch.norm(Q_manual - Q_auto).item()
        print(
            f"[step {step}] loss_diff={loss_diff:.3e}, grad_diff={grad_diff:.3e}, vec_diff={vec_diff:.3e}, Q_diff={Q_diff:.3e}"
        )

    print(f"\nTotal time manual: {manual_time:.3f}s")
    print(f"Total time autograd: {auto_time:.3f}s")
    print("final manual vecs:", manual_manager.vecs)
    print("final auto vecs:", auto_manager.vecs)
