import time
import torch

def generate_upper_triangular_pairs(k):
    pairs = []
    for i in range(k-1):
        for j in range(i+1, k):
            pairs.append((i, j))
    return pairs

def givens_rotation_matrix(k, i, j, theta, device=None):
    G = torch.eye(k, device=device)
    c = torch.cos(theta)
    s = torch.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G

def build_Q(k, thetas, pairs, device):
    Q = torch.eye(k, device=device)
    for theta, (p, q) in zip(thetas, pairs):
        G = givens_rotation_matrix(k, p, q, theta, device)
        Q = Q @ G
    return Q


def build_Q_2(k, thetas, pairs, device):
    Q = torch.eye(k, device=device)
    for theta, (p, q) in zip(thetas, pairs):
        ci = torch.cos(theta)
        si = torch.sin(theta)
        Qp, Qq = Q[:, p].clone(), Q[:, q]
        Q[:, p] = ci * Qp + si * Qq
        Q[:, q] = -si * Qp + ci * Qq

    return Q

@torch.no_grad()
def grad_thetas(thetas, pq, grad_Q):
    m = thetas.shape[0]
    k = grad_Q.shape[0]

    device = thetas.device
    dtype = thetas.dtype
    # 0, 1,  2,    ..., m-2,         m-1
    # I, G1, G1G2, ..., G1G2...Gm-2, G1G2...Gm-1
    P = [torch.eye(k, device=device, dtype=dtype)]
    for j in range(m-1):
        p, q = pq[j]

        Pj = P[-1].clone()
        c, s = torch.cos(thetas[j]), torch.sin(thetas[j])
        Qp_old = P[-1][:, p]
        Qq_old = P[-1][:, q]
        Pj[:, p] = c * Qp_old + s * Qq_old
        Pj[:, q] = -s * Qp_old + c * Qq_old
        P.append(Pj)

    # 0, 1,  2,      ..., m-2,         m-1
    # I, Gm, Gm-1Gm, ..., G3...Gm-1Gm, G2...Gm-1Gm
    S = [torch.eye(k, device=device, dtype=dtype)]
    for j in range(m-1):
        p, q = pq[m-1-j]

        Sj = S[-1].clone()
        c, s = torch.cos(thetas[m - 1 - j]), torch.sin(thetas[m - 1 - j])
        Qp_old = S[-1][p, :]
        Qq_old = S[-1][q, :]
        Sj[p, :] = c * Qp_old - s * Qq_old
        Sj[q, :] = s * Qp_old + c * Qq_old
        S.append(Sj)

    g = torch.zeros(m, device=device, dtype=dtype)
    for j in range(m):
        p, q = pq[j]
        cos_theta, sin_theta = torch.cos(thetas[j]), torch.sin(thetas[j])
        dG = torch.tensor([[-sin_theta, -cos_theta],
                        [ cos_theta, -sin_theta]], device=device, dtype=dtype)
        L = P[j][:,[p, q]]          # shape (k, 2)
        R = S[m-1-j][[p, q],:]        # shape (2, k)
        A = L @ dG @ R               # shape (k, k)
        g[j] = torch.sum(grad_Q * A)

    return g

@torch.no_grad()
def grad_Q(Q, X, W, epsX, epsW):
    XtX = X.t() @ X
    WWT = W @ W.t()
    epsXtX = epsX.t() @ epsX
    epsWDW = epsW @ epsW.t()
    
    grad = 2 * (XtX @ Q @ epsWDW + WWT @ Q @ epsXtX)
    grad += 2 * (X.t() @ epsX @ Q.t() @ W @ epsW.t() + W @ epsW.t() @ Q.t() @ X.t() @ epsX)
    grad += 2 * (X.t() @ epsX @ epsWDW + W @ epsW.t() @ epsXtX)
    return grad

@torch.no_grad()
def grad_func(k, thetas, pairs, X, W, eps_X, eps_W, device):
    Q = build_Q(k, thetas, pairs, device=device)
    gQ = grad_Q(Q, X, W, eps_X, eps_W)
    res = grad_thetas(thetas, pairs, gQ)
    return res

def loss_fn(Q, X, W, epsX, epsW):
    term = X @ Q @ epsW + epsX @ Q.t() @ W + epsX @ epsW
    return torch.norm(term, p='fro')**2

def check_gradients(theta, pairs, X, W, epsX, epsW, device, h=1e-5, tol=1e-3):
    """
    Compare manual gradient, autograd gradient, and finite-difference gradient.
    """
    theta = theta.clone().detach().requires_grad_(True)

    # autograd gradient
    Q = build_Q(W.shape[0], theta, pairs, device=device)
    L = loss_fn(Q, X, W, epsX, epsW)
    L.backward()
    grad_auto = theta.grad.detach().clone()

    # manual gradient
    grad_manual = grad_func(W.shape[0], theta.detach(), pairs, X, W, epsX, epsW, device=device)

    # finite-difference gradient
    grad_fd = torch.zeros_like(theta)
    for i in range(len(theta)):
        theta_p = theta.clone().detach()
        theta_m = theta.clone().detach()
        theta_p[i] += h
        theta_m[i] -= h
        L_p = loss_fn(build_Q(W.shape[0], theta_p, pairs, device=device), X, W, epsX, epsW)
        L_m = loss_fn(build_Q(W.shape[0], theta_m, pairs, device=device), X, W, epsX, epsW)
        grad_fd[i] = (L_p - L_m) / (2 * h)

    # report differences
    diff_manual = torch.norm(grad_manual - grad_fd) / torch.norm(grad_fd)
    diff_auto   = torch.norm(grad_auto   - grad_fd) / torch.norm(grad_fd)

    print(f"‣ relative error (manual vs fd): {diff_manual:.3e}")
    print(f"‣ relative error (auto   vs fd): {diff_auto:.3e}")
    return grad_manual, grad_auto, grad_fd

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"
    m, k, n = 2, 32, 32
    givens = 10
    update_steps = 10

    X = torch.randn(m, k, device=device)
    W = torch.randn(k, n, device=device)
    epsX = 0.01 * torch.randn(m, k, device=device)
    epsW = 0.01 * torch.randn(k, n, device=device)
    pairs = generate_upper_triangular_pairs(k)

    theta_manual = torch.nn.Parameter(0.3 * torch.ones(givens, device=device))
    theta_auto   = torch.nn.Parameter(0.3 * torch.ones(givens, device=device))

    opt_manual = torch.optim.SGD([theta_manual], lr=0.1)
    opt_auto   = torch.optim.SGD([theta_auto], lr=0.1)

    manual_time = 0.0
    auto_time = 0.0

    for step in range(update_steps):
        # ---- manual ----
        start_manual = time.time()
        opt_manual.zero_grad()
        Qm = build_Q(k, theta_manual, pairs, device=device)
        Qm2 = build_Q_2(k, theta_manual, pairs, device=device)
        print("Q diff:", torch.norm(Qm - Qm2).item())
        exit(0)
        L_manual = loss_fn(Qm, X, W, epsX, epsW)
        g_manual = grad_func(k, theta_manual, pairs, X, W, epsX, epsW, device=device)
        with torch.no_grad():
            theta_manual.grad = g_manual.clone()
        opt_manual.step()
        manual_time += time.time() - start_manual

        # ---- autograd ----
        start_auto = time.time()
        opt_auto.zero_grad()
        Qa = build_Q(k, theta_auto, pairs, device=device)
        L_auto = loss_fn(Qa, X, W, epsX, epsW)
        L_auto.backward()
        g_auto = theta_auto.grad.clone()
        opt_auto.step()
        auto_time += time.time() - start_auto

        # ---- diff ----
        loss_diff = torch.norm(L_manual.detach() - L_auto.detach()).item()
        grad_diff = torch.norm(g_manual - g_auto).item()
        theta_diff = torch.norm(theta_manual.detach() - theta_auto.detach()).item()
        print(f"[step {step}] loss_diff={loss_diff:.3e}, grad_diff={grad_diff:.3e}, theta_diff={theta_diff:.3e}")

        # ---- gradient check ----
        print("\nRunning gradient check...")
        grad_manual, grad_auto, grad_fd = check_gradients(
            theta_manual, pairs, X, W, epsX, epsW, device=device
        )

    print(f"\nTotal time manual: {manual_time:.3f}s")
    print(f"Total time autograd: {auto_time:.3f}s")
    print("final manual theta:", theta_manual)
    print("final auto theta:", theta_auto)
