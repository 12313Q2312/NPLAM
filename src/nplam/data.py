"""Deterministic data generation and leakage-free 60/20/20 partitions."""
from dataclasses import dataclass
import hashlib
import numpy as np


TASKS = {"mixed30": 30, "linear30": 30, "nonlinear30": 30, "dense30": 30, "sparse300": 300}


def components(x, task):
    c = np.zeros_like(x, dtype=np.float64)
    if task in {"mixed30", "sparse300"}:
        c[:, 0] = 4 * np.sin(2*x[:, 0])
        c[:, 1] = np.exp(x[:, 1])
        c[:, 2] = x[:, 2]**2
        c[:, 3] = -x[:, 3]
        c[:, 4] = 5*x[:, 4]
    elif task == "linear30":
        c[:, :5] = x[:, :5] * [2, 3, -2.5, -3.5, 4]
    elif task == "nonlinear30":
        c[:, 0] = 2*np.sin(2*x[:, 0])
        c[:, 1] = x[:, 1]**2
        c[:, 2] = np.exp(-x[:, 2])
        c[:, 3] = 3*np.cos(3*x[:, 3])
        c[:, 4] = -2*x[:, 4]**2
    elif task == "dense30":
        c[:, :15] = x[:, :15] * [2,3,-2.5,-3.5,4,-2,-3,2.5,3.5,-4,1.5,2.5,-2,-3,3.5]
        for j, a, b, e, d, q, freq in [(15,2,1,-1,3,-2,3),(20,-2,-1,-1,-3,2,3),(25,3,1.5,1.5,4,1,2)]:
            c[:, j] = a*np.sin(2*x[:, j])
            c[:, j+1] = b*x[:, j+1]**2
            c[:, j+2] = e*np.exp(-x[:, j+2])
            c[:, j+3] = d*np.cos(freq*x[:, j+3])
            c[:, j+4] = q*x[:, j+4]**2
    else:
        raise ValueError(task)
    return c


def synthetic(task, n=1600, seed=0, noise=1.0):
    if task not in TASKS or n < 10 or noise < 0:
        raise ValueError("invalid synthetic task, n, or noise")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, (n, TASKS[task]))
    c = components(x, task)
    y = c.sum(axis=1) + rng.normal(0, noise, n)
    truth = np.zeros(TASKS[task], dtype=int)
    if task in {"mixed30", "sparse300"}:
        truth[:5] = [2,2,2,1,1]
    elif task == "linear30":
        truth[:5] = 1
    elif task == "nonlinear30":
        truth[:5] = 2
    else:
        truth[:] = [1]*15 + [2]*15
    return x, y, truth


class TrainMinMax:
    """Train-fitted affine transform to [-1,1]; never clip held-out extremes."""
    def fit(self, x):
        self.low = np.min(x, axis=0)
        high = np.max(x, axis=0)
        self.scale = np.where(high-self.low > 1e-12, (high-self.low)/2, 1.0)
        self.center = (high+self.low)/2
        return self

    def transform(self, x):
        return ((x-self.center)/self.scale).astype(np.float32)

    def inverse(self, x):
        return x*self.scale + self.center

    def state(self):
        return {k: np.asarray(getattr(self, k)).tolist() for k in ["low", "center", "scale"]}


@dataclass
class Split:
    x: list
    y: list
    indices: list
    x_scaler: TrainMinMax
    y_scaler: TrainMinMax
    fingerprint: str


def split_scale(x, y, seed=0, chronological=False):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite numeric matching x/y with at least 10 rows required")
    idx = np.arange(len(y)) if chronological else np.random.default_rng(seed).permutation(len(y))
    indices = list(np.split(idx, [int(.6*len(y)), int(.8*len(y))]))
    xs, ys = TrainMinMax().fit(x[indices[0]]), TrainMinMax().fit(y[indices[0]])
    digest = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
    return Split([xs.transform(x[i]) for i in indices], [ys.transform(y[i]) for i in indices], indices, xs, ys, digest)
