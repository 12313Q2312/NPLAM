"""Scientific artifacts (PNG and vector SVG), no interactive dependency."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import TASKS, components


def save(fig, target):
    fig.tight_layout()
    fig.savefig(str(target)+".png", dpi=140)
    fig.savefig(str(target)+".svg")
    plt.close(fig)


def diagnostics(model, split, y, pred, history, task, target):
    target = Path(target)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(y,pred,s=5,alpha=.35)
    lo,hi = min(y.min(),pred.min()),max(y.max(),pred.max())
    ax.plot([lo,hi],[lo,hi],color="black",linewidth=1)
    ax.set(xlabel="Observed (original units)",ylabel="Predicted",title=task)
    save(fig,target/"predictions")
    if history:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot([h["epoch"] for h in history],[h["validation_mse"] for h in history])
        ax.set(xlabel="Epoch",ylabel="Validation MSE (scaled)",yscale="log")
        save(fig,target/"learning")
    if hasattr(model,"g1"):
        fig,axes = plt.subplots(2,1,figsize=(7,5))
        for ax,key in zip(axes,["g1","g2"]):
            values=np.asarray([h[key] for h in history])
            ax.plot(values,alpha=.4,linewidth=.6)
            ax.set(ylabel=key,xlabel="Epoch")
        save(fig,target/"gates")
    if hasattr(model,"contributions"):
        k = min(6, split.x[0].shape[1])
        fig,axes = plt.subplots(2,3,figsize=(11,6))
        grid = np.linspace(-1,1,201,dtype=np.float32)
        for j,ax in enumerate(axes.flat):
            if j>=k:
                ax.axis("off")
                continue
            x=np.zeros((len(grid),split.x[0].shape[1]),np.float32)
            x[:,j]=grid
            with torch.no_grad():
                curve=model.contributions(torch.from_numpy(x))[:,j].numpy()*split.y_scaler.scale
            curve-=curve.mean()
            raw=split.x_scaler.inverse(x)
            ax.plot(raw[:,j],curve,label="Learned (grid-centered)")
            if task in TASKS:
                true=components(raw,task)[:,j]
                ax.plot(raw[:,j],true-true.mean(),"--",label="Truth (grid-centered)")
            ax.set(title=f"Feature {j+1}",xlabel="Original input",ylabel="Contribution")
        axes.flat[0].legend(fontsize=7)
        save(fig,target/"contributions")
