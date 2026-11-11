from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Lasso
from .data import synthetic, split_scale, TASKS
from .metrics import regression, recovery
from .model import NPLAM
from .trainer import seed_everything, build_model, fit_neural, predict
from .real_data import load_real
from .plots import diagnostics


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git",*args],text=True).strip()


def environment():
    try:
        sha, dirty = git("rev-parse","HEAD"), bool(git("status","--porcelain","--untracked-files=no"))
    except (subprocess.CalledProcessError,FileNotFoundError):
        sha, dirty = "unavailable", None
    sources={str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("src").rglob("*.py"))}
    return {"source_sha":sha,"tracked_dirty":dirty,"source_hashes":sources,"python":sys.version,
            "platform":platform.platform(),"processor":platform.processor(),"device":"cpu","threads":1,
            "packages":{p:importlib.metadata.version(p) for p in ["torch","numpy","scikit-learn","pandas","matplotlib","PyYAML"]}}


def compatible_reuse(config, source):
    """Refuse reusing scientific results after a change to their method/settings."""
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    if manifest["config"] != config:
        raise ValueError("reuse configuration differs")
    current = environment()
    required = ["data.py", "model.py", "trainer.py", "metrics.py"]
    if any(t["name"] not in TASKS for t in config["tasks"]):
        required.append("real_data.py")
    for name in required:
        key = "src/nplam/" + name
        if current["source_hashes"][key] != manifest["environment"]["source_hashes"][key]:
            raise ValueError(f"reuse method changed: {name}")
    if current["packages"] != manifest["environment"]["packages"]:
        raise ValueError("reuse package versions differ")
    return manifest


def reusable_result(source, target, task, seed, label, fingerprint):
    source = source / f"{task}-seed{seed}"
    folder = source / label
    required = ["metrics.json", "selection.json", "history.json", "predictions.npz"]
    model_file = "model.json" if label == "lasso" else "model.pt"
    if not all((folder/name).is_file() for name in required+[model_file]):
        return None
    metadata = json.loads((source/"preprocessing.json").read_text(encoding="utf-8"))
    if metadata["fingerprint"] != fingerprint:
        raise ValueError("reuse data fingerprint differs")
    row = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
    if (row["task"], row["seed"], row["model"]) != (task, seed, label):
        raise ValueError("reuse run identity differs")
    with np.load(folder/"predictions.npz", allow_pickle=False) as preds:
        measured = regression(preds["y"], preds["pred"])
    if not np.isclose(measured["mse"], row["mse"], rtol=1e-10):
        raise ValueError("reuse predictions disagree with metrics")
    shutil.copytree(folder, target)
    return row


def run(config, output, cache="data", command=None, reuse_from=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"manifest.json").exists():
        raise FileExistsError("output already has a manifest; choose a fresh output directory")
    previous = compatible_reuse(config, Path(reuse_from)) if reuse_from else None
    manifest = {"config":config,"environment":environment(),"command":command or sys.argv,
                "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), "reused_runs":[]}
    if previous:
        manifest["reused_from"] = {"path":str(reuse_from),"source_sha":previous["environment"]["source_sha"],
            "manifest_sha256":hashlib.sha256((Path(reuse_from)/"manifest.json").read_bytes()).hexdigest()}
    dump(output/"manifest.json",manifest)
    rows=[]
    for task_spec in config["tasks"]:
        task=task_spec["name"]
        for seed in config["seeds"]:
            seed_everything(seed)
            subset_indices = None
            download_manifest = None
            if task in TASKS:
                x,y,truth=synthetic(task,task_spec.get("samples",1600),seed)
                names=[f"x{i+1}" for i in range(x.shape[1])]
            else:
                x,y,names=load_real(task,cache)
                download_manifest = json.loads((Path(cache)/(task+".json")).read_text(encoding="utf-8"))
                truth=None
                cap=task_spec.get("samples")
                if cap and len(y)>cap:
                    # Temporal thinning for Beijing; uniform subsample before random splits otherwise.
                    idx=np.linspace(0,len(y)-1,cap,dtype=int) if task=="beijing" else np.sort(np.random.default_rng(seed).choice(len(y),cap,replace=False))
                    x,y=x[idx],y[idx]
                    subset_indices = idx.tolist()
            split=split_scale(x,y,seed,chronological=task=="beijing")
            data_dir=output/f"{task}-seed{seed}"
            data_dir.mkdir()
            np.savez_compressed(data_dir/"split.npz",train=split.indices[0],validation=split.indices[1],test=split.indices[2])
            dump(data_dir/"preprocessing.json",{"x":split.x_scaler.state(),"y":split.y_scaler.state(),
                "features":names,"fingerprint":split.fingerprint,"samples":len(y),"data_seed":seed,"split_seed":seed,
                "truth":truth.tolist() if truth is not None else None,
                "source_data":download_manifest,"subset_indices_in_clean_data":subset_indices})
            for spec in config["models"]:
                label=spec.get("label",spec["name"])
                target=data_dir/label
                reused = reusable_result(Path(reuse_from), target, task, seed, label, split.fingerprint) if previous else None
                if reused is not None:
                    dump(target/"provenance.json",manifest["reused_from"])
                    rows.append(reused)
                    manifest["reused_runs"].append({"task":task,"seed":seed,"model":label})
                    dump(output/"manifest.json",manifest)
                    pd.DataFrame(rows).to_csv(output/"results.csv",index=False)
                    print(f"REUSED {task} seed={seed} {label}",flush=True)
                    continue
                target.mkdir()
                start=time.perf_counter()
                candidates=[]
                best=None
                for hyper in spec["grid"]:
                    seed_everything(seed)
                    if spec["name"]=="lasso":
                        model=Lasso(alpha=hyper["alpha"],max_iter=20000,tol=1e-6)
                        t=time.perf_counter()
                        model.fit(split.x[0],split.y[0])
                        val=float(np.mean((model.predict(split.x[1])-split.y[1])**2))
                        info={"validation_mse":val,"fit_seconds":time.perf_counter()-t,"history":[]}
                    else:
                        model=build_model(spec["name"],x.shape[1],config["hidden"],spec.get("mode","paper"),spec.get("ablation","none"))
                        model,info=fit_neural(model,split,seed,config["epochs"],hyper["lr"],hyper.get("lambdas",[0,0,0]),
                                              config["batch_size"],config["patience"])
                    candidates.append({"hyperparameters":hyper,**{k:v for k,v in info.items() if k!="history"}})
                    if best is None or info["validation_mse"]<best[2]["validation_mse"]:
                        best=(model,hyper,info)
                model,hyper,info=best
                pred=model.predict(split.x[2]) if spec["name"]=="lasso" else predict(model,split.x[2])
                raw_y=split.y_scaler.inverse(split.y[2])
                raw_pred=split.y_scaler.inverse(pred)
                stats=regression(raw_y,raw_pred)
                row={"task":task,"model":label,"seed":seed,**stats,
                     "mse_scaled":float(np.mean((pred-split.y[2])**2)),"validation_mse":info["validation_mse"],
                     "search_seconds":time.perf_counter()-start,"fit_seconds":info["fit_seconds"]}
                if spec["name"]=="lasso":
                    structure=np.where(np.abs(model.coef_)>config["fs_threshold"],1,0)
                    row.update(parameters=int(model.coef_.size+1),trainable_parameters=int(model.coef_.size+1))
                    dump(target/"model.json",{"coef":model.coef_.tolist(),"intercept":float(model.intercept_)})
                else:
                    row.update(parameters=sum(p.numel() for p in model.parameters())+sum(b.numel() for b in model.buffers()),
                               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
                    torch.save({"state_dict":model.state_dict(),"spec":spec,"features":x.shape[1],"hidden":config["hidden"]},target/"model.pt")
                    structure=model.structure(config["fs_threshold"],config["sd_threshold"]) if isinstance(model,NPLAM) else None
                    if isinstance(model,NPLAM):
                        dump(target/"gates.json",{"g1":model.g1.detach().tolist(),"g2":model.g2.detach().tolist(),
                            "effective_linear_weights":model.effective_linear_weights().detach().tolist(),"structure":structure.tolist()})
                if truth is not None and structure is not None:
                    row.update(recovery(truth,structure))
                np.savez_compressed(target/"predictions.npz",y=raw_y,pred=raw_pred)
                dump(target/"history.json",info["history"])
                dump(target/"selection.json",{"candidates":candidates,"selected":hyper,"best_epoch":info.get("best_epoch"),
                     "selection":"minimum validation MSE; test evaluated only after selection"})
                dump(target/"metrics.json",row)
                if seed==config["seeds"][0] and config.get("plots",True) and spec["name"]!="lasso":
                    diagnostics(model,split,raw_y,raw_pred,info["history"],task,target)
                rows.append(row)
                pd.DataFrame(rows).to_csv(output/"results.csv",index=False)
                print(f"{task} seed={seed} {label}: mse={stats['mse']:.5g} f1={row.get('f1')} elapsed={row['search_seconds']:.1f}s",flush=True)
    summarize(output)
    manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    manifest["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    manifest["runs_completed"]=len(rows)
    dump(output/"manifest.json",manifest)


def summarize(output):
    output = Path(output)
    frame = pd.read_csv(output / "results.csv")
    keys = ["task", "model"]
    if "suite" in frame:
        keys.insert(0, "suite")
    candidates = [
        "mse", "rmse", "mae", "mse_scaled", "r2", "tp", "fp", "tn", "fn", "f1",
        "cf", "uf", "of", "missed_structure", "three_class_accuracy",
        "search_seconds", "fit_seconds", "parameters", "trainable_parameters",
    ]
    metrics = [key for key in candidates if key in frame]
    groups = frame.groupby(keys)
    grouped = groups[metrics].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.to_csv(output/"summary.csv")
    labels = ["任务", "模型", "种子数", "MSE", "R²", "F1", "CF"]
    if "suite" in frame:
        labels.insert(0, "实验组")
    lines = [
        "# 实验自动汇总", "",
        "每行报告均值与样本标准差（ddof=1）。MSE 使用原始目标的平方单位。",
        "", "|" + "|".join(labels) + "|", "|" + "|".join(["---"] * len(labels)) + "|",
    ]
    for identity, group in groups:
        values = []
        for metric in ["mse", "r2", "f1", "cf"]:
            measured = group[metric].dropna() if metric in group else []
            if len(measured) > 1:
                values.append(f"{measured.mean():.5g} ± {measured.std():.3g}")
            elif len(measured) == 1:
                values.append(f"{measured.iloc[0]:.5g}")
            else:
                values.append("N/A")
        cells = [*identity, str(len(group)), *values]
        lines.append("|" + "|".join(cells) + "|")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
