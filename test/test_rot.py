import torch

from segquant.utils.cayley_optimizer import CayleySGD

def loss_func(Q, X, W, epsX, epsW):
    # L(Q) = || X Q eps_W + eps_X Q^T W + eps_X eps_W ||_F^2
    term1 = X @ Q @ epsW
    term2 = epsX @ Q.t() @ W
    term3 = epsX @ epsW
    loss = torch.norm(term1 + term2 + term3, p='fro') ** 2
    return loss

def grad_func(Q, X, W, epsX, epsW):
    grad = 2 * X.t() @ X @ Q @ (epsW @ epsW.t())
    grad += 2 * W @ W.t() @ Q @ (epsX.t() @ epsX)
    grad += 2 * X.t() @ epsX @ Q.t() @ W @ epsW.t()
    grad += 2 * W @ epsW.t() @ Q.t() @ X.t() @ epsX
    grad += 2 * X.t() @ epsX @ (epsW @ epsW.t())
    grad += 2 * W @ epsW.t() @ (epsX.t() @ epsX)
    return grad

if __name__ == "__main__":
    in_features = 50
    out_features = 100
    batch = 2

    X = torch.randn(batch, in_features)
    W = torch.randn(out_features, in_features).T
    epsX = 0.01 * torch.randn(batch, in_features)
    epsW = 0.01 * torch.randn(out_features, in_features).T

    Q0, _ = torch.linalg.qr(torch.randn(in_features, in_features))

    def gradQ(Q):
        return grad_func(Q, X, W, epsX, epsW)
    def lossQ(Q):
        return loss_func(Q, X, W, epsX, epsW)

    optimizer = CayleySGD(gradQ, lr=0.1)
    Q = Q0.clone()
    for i in range(1000):
        Q = optimizer.step(Q)
        print(f"Step {i}: loss = {lossQ(Q).item():.4e}, orth_error = {torch.norm(Q @ Q.t() - torch.eye(in_features)).item():.4e}")
