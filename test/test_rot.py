import torch

from segquant.layers.givens_orthogonal import GivensOrthogonal


if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    device = "cpu"
    m, k, n = 2, 32, 32
    givens = 10
    update_steps = 10
    sample_mode = "rand"
    init_mode = "random"

    grad = "buffer"  # 'buffer', 'normal', 'slow'

    if grad == "buffer":
        enable_grad_buffer = True
        enable_low_memory_grad = False
    elif grad == "normal":
        enable_grad_buffer = False
        enable_low_memory_grad = False
    elif grad == "slow":
        enable_grad_buffer = False
        enable_low_memory_grad = True
    else:
        raise ValueError("Unknown grad mode")

    print(
        f"Testing GivensManager with k={k}, givens={givens}, sample_mode={sample_mode}, init_mode={init_mode}, grad={grad}"
    )

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
        grad += 2 * (
            X.t() @ epsX @ Q.t() @ W @ epsW.t() + W @ epsW.t() @ Q.t() @ X.t() @ epsX
        )
        grad += 2 * (X.t() @ epsX @ epsWDW + W @ epsW.t() @ epsXtX)
        return grad

    @torch.no_grad()
    def manual_grad(manager, X, W, epsX, epsW):
        Q = manager.forward()
        gQ = manual_grad_Q(Q, X, W, epsX, epsW)
        g = manager.try_grad(chain_grad=gQ)
        return g

    manual_manager = GivensOrthogonal(
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

    auto_manager = GivensOrthogonal(
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
