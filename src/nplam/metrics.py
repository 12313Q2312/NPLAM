import numpy as np


def regression(y, pred):
    y, pred = np.asarray(y).reshape(-1), np.asarray(pred).reshape(-1)
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("invalid predictions")
    mse = float(np.mean((y-pred)**2))
    return {"mse": mse, "rmse": mse**.5, "mae": float(np.abs(y-pred).mean()),
            "r2": float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)) if np.var(y)>0 else None}


def recovery(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or not np.isin(predicted, [0,1,2]).all():
        raise ValueError("invalid structure labels")
    active, chosen = truth > 0, predicted > 0
    tp, fp = int((active & chosen).sum()), int((~active & chosen).sum())
    tn, fn = int((~active & ~chosen).sum()), int((active & ~chosen).sum())
    # Denominator includes every true active feature, including missed features.
    den = max(1, int(active.sum()))
    return {"tp":tp,"fp":fp,"tn":tn,"fn":fn,"f1":2*tp/max(1,2*tp+fp+fn),
            "cf":float(((truth==predicted)&active).sum()/den),
            "uf":float(((truth==2)&(predicted==1)).sum()/den),
            "of":float(((truth==1)&(predicted==2)).sum()/den),
            "missed_structure":float((active & ~chosen).sum()/den),
            "three_class_accuracy":float((truth==predicted).mean())}
