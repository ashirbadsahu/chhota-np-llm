from arr import xp

class AdamOptimizer:
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, max_grad_norm=1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.max_grad_norm = max_grad_norm

        self.m = {}
        self.v = {}

        for name, param in params:
            self.m[name] = xp.zeros_like(param)
            self.v[name] = xp.zeros_like(param)

    def _clip_grads(self, grads):
        total_norm = 0.0
        for _, grad in grads:
            total_norm += xp.sum(grad ** 2)
        total_norm = float(xp.sqrt(total_norm))

        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            grads = [(n, g * clip_coef) for n, g in grads]

        return grads

    def step(self, params, grads):
        self.t += 1

        if self.max_grad_norm > 0:
            grads = self._clip_grads(grads)

        for (name, param), (_, grad) in zip(params, grads):
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)
