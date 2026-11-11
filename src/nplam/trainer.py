"""Validation-only model selection; bounded, deterministic CPU experiments."""
from copy import deepcopy
import random
import time
import numpy as np
import torch
from torch import nn
from .model import NPLAM, NAM


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def build_model(name, features, hidden, mode="paper", ablation="none"):
    if name == "nplam":
        return NPLAM(features, hidden, mode, ablation)
    if name == "nam":
        return NAM(features, hidden)
    if name == "mlp":
        return nn.Sequential(nn.Linear(features,25), nn.ReLU(),nn.Linear(25,25),nn.ReLU(),nn.Linear(25,1),nn.Flatten(0))
    raise ValueError(name)


@torch.no_grad()
def predict(model, x, batch_size=1024):
    return torch.cat([model(torch.as_tensor(x[i:i+batch_size], dtype=torch.float32)).reshape(-1)
                      for i in range(0,len(x),batch_size)]).cpu().numpy()


def fit_neural(model, split, seed, epochs, lr, lambdas=(0,0,0), batch_size=256, patience=80):
    x, y = torch.from_numpy(split.x[0]), torch.from_numpy(split.y[0])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    generator = torch.Generator().manual_seed(seed+12345)
    best, state, best_epoch, stale = float("inf"), deepcopy(model.state_dict()), 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x), generator=generator).split(batch_size):
            opt.zero_grad(set_to_none=True)
            loss = ((model(x[idx]).reshape(-1)-y[idx])**2).mean()
            if hasattr(model, "regularization"):
                loss = loss + model.regularization(lambdas)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            opt.step()
        model.eval()
        val = float(np.mean((predict(model,split.x[1])-split.y[1])**2))
        row = {"epoch":epoch+1,"validation_mse":val}
        if isinstance(model, NPLAM):
            row.update(g1=model.g1.detach().tolist(),g2=model.g2.detach().tolist())
        history.append(row)
        if val < best - 1e-10:
            best, state, best_epoch, stale = val, deepcopy(model.state_dict()), epoch+1, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    return model, {"best_epoch":best_epoch,"validation_mse":best,"epochs_run":len(history),
                   "fit_seconds":time.perf_counter()-start,"history":history}
