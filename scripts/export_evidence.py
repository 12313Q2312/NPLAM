"""Validate completed benchmark artifacts and export compact review evidence."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nplam.experiment import dump, summarize
from nplam.metrics import regression


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_job(root, outcome):
    job = outcome["job"]
    if outcome["exit_code"]:
        raise ValueError(f"{job} failed")
    directory = root / job
    manifest = read_json(directory / "manifest.json")
    config = manifest["config"]
    models = [spec.get("label", spec["name"]) for spec in config["models"]]
    expected = set(itertools.product(
        [task["name"] for task in config["tasks"]], config["seeds"], models,
    ))
    rows = pd.read_csv(directory / "results.csv")
    observed = set(rows[["task", "seed", "model"]].itertuples(index=False, name=None))
    if expected != observed or len(rows) != len(expected) or manifest["runs_completed"] != len(rows):
        raise ValueError(f"{job} does not contain exactly the configured runs")

    for row in rows.itertuples():
        run_dir = directory / f"{row.task}-seed{row.seed}" / row.model
        model_file = "model.json" if row.model == "lasso" else "model.pt"
        for filename in (model_file, "selection.json", "history.json", "metrics.json"):
            if not (run_dir / filename).is_file():
                raise FileNotFoundError(run_dir / filename)
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as predictions:
            computed = regression(predictions["y"], predictions["pred"])
        if not np.isclose(computed["mse"], row.mse, rtol=1e-10):
            raise ValueError(f"prediction MSE mismatch in {run_dir}")

    rows.insert(0, "suite", job.split("-", 1)[0])
    return rows, manifest


def plot_results(rows, directory):
    tasks = ["mixed30", "linear30", "nonlinear30", "dense30", "sparse300"]
    baseline = rows[rows["suite"].isin(["synthetic", "highdim"])]
    markers = {"nplam": "o", "nam": "s", "mlp": "D", "lasso": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    positions = np.arange(len(tasks))
    for index, (model, marker) in enumerate(markers.items()):
        by_task = baseline[baseline["model"] == model].groupby("task")["mse"]
        means = by_task.mean().reindex(tasks)
        deviations = by_task.std().reindex(tasks)
        lower_error = np.minimum(deviations, means * 0.99)
        axes[0].errorbar(
            positions + (index - 1.5) * 0.11, means,
            yerr=[lower_error, deviations], fmt=marker, capsize=3, label=model.upper(),
        )
    axes[0].set(yscale="log", ylabel="Test MSE (original scale)", title="Synthetic prediction")
    axes[0].legend(frameon=False, fontsize=8)
    selected = baseline[baseline["model"] == "nplam"].groupby("task")
    for index, (metric, marker) in enumerate([("f1", "o"), ("cf", "s")]):
        axes[1].errorbar(
            positions + (index - 0.5) * 0.13, selected[metric].mean().reindex(tasks),
            yerr=selected[metric].std().reindex(tasks), fmt=marker, capsize=3, label=metric.upper(),
        )
    axes[1].set(ylabel="Fraction", title="NPLAM structure recovery", ylim=(-0.08, 1.12))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(positions, tasks, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Synthetic tasks: mean and sample SD (30D: 10 seeds; 300D: 3 seeds)")
    for extension in ["png", "svg"]:
        fig.savefig(directory / f"synthetic-summary.{extension}", dpi=150)
    plt.close(fig)


def export(root, output):
    outcomes = read_json(root / "jobs.json")
    jobs = {entry["job"] for entry in outcomes}
    expected_jobs = {
        "synthetic-mixed30", "synthetic-linear30", "synthetic-nonlinear30",
        "synthetic-dense30", "highdim-sparse300", "ablations-mixed30",
        "real-california", "real-superconductivity", "real-beijing",
    }
    if jobs != expected_jobs or len(jobs) != len(outcomes):
        raise ValueError("benchmark must contain the nine configured jobs exactly once")
    verified = [validate_job(root, outcome) for outcome in outcomes]
    output.mkdir(parents=True)
    figures = output / "figures"
    manifests = output / "manifests"
    figures.mkdir()
    manifests.mkdir()
    rows = pd.concat([frame for frame, _ in verified], ignore_index=True)
    rows.to_csv(output / "results.csv", index=False)
    summarize(output)
    dump(output / "jobs.json", outcomes)
    for outcome, (_, manifest) in zip(outcomes, verified):
        job = outcome["job"]
        dump(manifests / f"{job}.json", manifest)
        for task in manifest["config"]["tasks"]:
            name = task["name"]
            if job.startswith("real-"):
                seed = manifest["config"]["seeds"][0]
                prep = read_json(root / job / f"{name}-seed{seed}" / "preprocessing.json")
                dump(manifests / f"data-{name}.json", prep["source_data"])

    example = root / "synthetic-mixed30" / "mixed30-seed10" / "nplam"
    for stem in ["contributions", "gates", "learning", "predictions"]:
        for extension in ["png", "svg"]:
            source = example / f"{stem}.{extension}"
            shutil.copyfile(source, figures / source.name)
    plot_results(rows, figures)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    hashes = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}" for path in files]
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(f"Verified {len(rows)} runs across {len(outcomes)} jobs and exported to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.benchmark, args.output)
