"""Models with closed-form, vectorised per-sample gradients (needed for DP-SGD)."""
import numpy as np


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogReg:
    name = "logreg"

    def __init__(self, d):
        self.d_in = d
        self.dim = d + 1

    def init(self, rng):
        return np.zeros(self.dim)

    def predict(self, w, X):
        return _sig(X @ w[:-1] + w[-1])

    def per_sample_grad(self, w, X, y):
        r = (self.predict(w, X) - y)[:, None]
        return np.hstack([r * X, r])

    def loss(self, w, X, y):
        p = np.clip(self.predict(w, X), 1e-7, 1 - 1e-7)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class MLP:
    """One hidden layer (tanh) binary classifier."""
    name = "mlp"

    def __init__(self, d, h=32):
        self.d_in, self.h = d, h
        self.dim = d * h + h + h + 1

    def _unpack(self, w):
        d, h = self.d_in, self.h
        W1 = w[:d * h].reshape(d, h); b1 = w[d * h:d * h + h]
        W2 = w[d * h + h:d * h + 2 * h]; b2 = w[-1]
        return W1, b1, W2, b2

    def init(self, rng):
        d, h = self.d_in, self.h
        w = np.zeros(self.dim)
        w[:d * h] = rng.normal(0, 1 / np.sqrt(d), d * h)
        w[d * h + h:d * h + 2 * h] = rng.normal(0, 1 / np.sqrt(h), h)
        return w

    def predict(self, w, X):
        W1, b1, W2, b2 = self._unpack(w)
        return _sig(np.tanh(X @ W1 + b1) @ W2 + b2)

    def per_sample_grad(self, w, X, y):
        W1, b1, W2, b2 = self._unpack(w)
        H = np.tanh(X @ W1 + b1)
        r = (_sig(H @ W2 + b2) - y)[:, None]               # n x 1
        gW2 = r * H                                        # n x h
        dH = r * W2[None, :] * (1 - H ** 2)                # n x h
        gW1 = (X[:, :, None] * dH[:, None, :]).reshape(len(X), -1)
        return np.hstack([gW1, dH, gW2, r])

    def loss(self, w, X, y):
        p = np.clip(self.predict(w, X), 1e-7, 1 - 1e-7)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class ExpPH:
    """Exponential proportional-hazards model: hazard = exp(x.w + b). y = [event, time].
    Per-record negative log-likelihood  l_i = lambda_i T_i - E_i log(lambda_i)  is decomposable,
    so DP-SGD clipping/accounting apply exactly (unlike the Cox partial likelihood)."""
    name = "expph"

    def __init__(self, d):
        self.d_in, self.dim = d, d + 1

    def init(self, rng):
        return np.zeros(self.dim)

    def predict(self, w, X):           # risk score (higher = shorter survival)
        return X @ w[:-1] + w[-1]

    def per_sample_grad(self, w, X, y):
        lam = np.exp(np.clip(self.predict(w, X), -20, 20))
        r = (lam * y[:, 1] - y[:, 0])[:, None]
        return np.hstack([r * X, r])

    def loss(self, w, X, y):
        eta = np.clip(self.predict(w, X), -20, 20)
        return np.exp(eta) * y[:, 1] - y[:, 0] * eta


def make_model(name, d):
    if name == "expph":
        return ExpPH(d)
    return LogReg(d) if name == "logreg" else MLP(d)
