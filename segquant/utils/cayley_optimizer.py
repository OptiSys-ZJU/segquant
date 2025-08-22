import torch

epsilon = 1e-8

def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out

class CayleyOptimizer:
    def __init__(self, q=0.5, s=5):
        self.q = q
        self.s = s

    def _loop(self, X, W, M, alpha):
        Y = X + alpha * M
        for _ in range(self.s):
            Y = X + 0.5 * alpha * torch.matmul(W, X + Y)
        return Y


class CayleySGD(CayleyOptimizer):
    def __init__(
        self,
        grad_func,
        lr=0.01,
        momentum=0,
        dampening=0,
        weight_decay=0,
        q=0.5,
        s=5,
    ):
        super().__init__(q, s)
        self.grad_func = grad_func
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay

        self.M = 0

    def step(self, param: torch.Tensor):
        grad = self.grad_func(param)
        if self.weight_decay != 0:
            grad.add_(
                self.weight_decay, param
            )  # grad = grad + weight_decay * param

        self.M = self.momentum * self.M + (1 - self.dampening) * grad

        Wk_hat = (
            self.M @ param.t()
            - 0.5 * param @ param.t() @ self.M @ param.t()
        )
        Wk = Wk_hat - Wk_hat.t()

        alpha = min(self.q * 2 / (matrix_norm_one(Wk) + epsilon), self.lr)
        param_next = self._loop(param, Wk, self.M, -alpha)

        self.M = Wk @ param

        return param_next


class CayleyAdam(CayleyOptimizer):
    def __init__(
        self,
        grad_func,
        lr=0.01,
        betas=(0.9, 0.999),
        weight_decay=0,
        q=0.5,
        s=5,
    ):
        super().__init__(q, s)
        self.grad_func = grad_func
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.betas_power = (self.betas[0], self.betas[1])
        self.M = 0
        self.v = 1

    def step(self, param: torch.Tensor):
        beta1, beta2 = self.betas

        grad = self.grad_func(param)
        if self.weight_decay != 0:
            grad.add_(self.weight_decay, param)

        self.M = beta1 * self.M + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (torch.norm(grad) ** 2)

        M_hat = self.M / (1 - self.betas_power[0])
        v_hat = self.v / (1 - self.betas_power[1])

        W_hat = M_hat @ param.t() - 0.5 * param @ param.t() @ M_hat @ param.t()
        W = (W_hat - W_hat.t()) / v_hat.add(epsilon).sqrt()

        alpha = min(self.q * 2 / (matrix_norm_one(W) + epsilon), self.lr)
        param_next = self._loop(param, W, self.M, -alpha)

        r = (1 - self.betas_power[0]) * v_hat.add(epsilon).sqrt()
        self.M = r * W @ param
        self.betas_power = (
            beta1 * self.betas_power[0],
            beta2 * self.betas_power[1],
        )

        return param_next


if __name__ == "__main__":
    p, n = 5, 50
    param = torch.randn(p, n)
    paramX, _ = torch.linalg.qr(param)


    def loss_func(Q):
        return torch.norm(Q - torch.eye(p)) ** 2

    def grad_func(Q):
        p, n = Q.shape
        return -torch.eye(p, n)

    optimizer = CayleySGD(grad_func, lr=0.5)
    param = paramX.clone()
    for i in range(100):
        # print(param)
        param = optimizer.step(param)
        print(f"Step {i}: loss = {loss_func(param).item():.4e}, orth_error = {torch.norm(param @ param.t() - torch.eye(p)).item():.4e}")

    print("=====================================")

    optimizer = CayleyAdam(grad_func, lr=0.5)
    param = paramX.clone()
    for i in range(100):
        # print(param)
        param = optimizer.step(param)
        print(f"Step {i}: loss = {loss_func(param).item():.4e}, orth_error = {torch.norm(param @ param.t() - torch.eye(p)).item():.4e}")
