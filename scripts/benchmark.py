"""Run fixed presets in up to four independent single-thread CPU processes."""
import concurrent.futures
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
import pandas as pd
from nplam.experiment import summarize, dump


def aggregate(root, outcomes):
    frames = []
    for outcome in outcomes:
        job = outcome["job"]
        results = root / job / "results.csv"
        if results.is_file():
            frame = pd.read_csv(results)
            frame.insert(0, "suite", job.split("-", 1)[0])
            frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(root / "results.csv", index=False)
        summarize(root)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("root",nargs="?",default="runs/benchmark")
    parser.add_argument("--reuse-root")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--summarize-only", action="store_true", help="regenerate combined reports from existing jobs")
    args=parser.parse_args()
    root=Path(args.root)
    if args.summarize_only:
        outcomes = json.loads((root / "jobs.json").read_text(encoding="utf-8"))
        aggregate(root, outcomes)
        return
    root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for name in ["synthetic","highdim","ablations","real"]:
        config=yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
        groups=[[task] for task in config["tasks"]] if name in {"synthetic","real"} else [config["tasks"]]
        for group in groups:
            key=name+"-"+group[0]["name"]
            cfg=dict(config,tasks=group)
            file=root/(key+".yaml")
            file.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
            jobs.append((key,file))
    def launch(job):
        key,file=job
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLCONFIGDIR=str((root/"mpl"/key).resolve()))
        cmd=[sys.executable,"-m","nplam.cli","run","--config",str(file),"--output",str(root/key)]
        previous=Path(args.reuse_root)/key if args.reuse_root else None
        if previous and (previous/"manifest.json").exists():
            cmd += ["--reuse-from",str(previous)]
        print("START "+key,flush=True)
        with (root/(key+".log")).open("w",encoding="utf-8") as log:
            p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        print(f"END {key}: {p.returncode}",flush=True)
        return {"job":key,"exit_code":p.returncode,"command":cmd}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes=list(pool.map(launch,jobs))
    dump(root/"jobs.json",outcomes)
    aggregate(root, outcomes)
    if any(j["exit_code"] for j in outcomes):
        raise SystemExit("One or more jobs failed; inspect jobs.json and logs. Partial results are preserved.")


if __name__=="__main__":
    main()
