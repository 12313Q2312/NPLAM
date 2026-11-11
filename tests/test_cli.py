import json
import subprocess
import sys
import pandas as pd
from nplam.experiment import summarize


def test_summary_does_not_pool_ablation_and_primary_seeds(tmp_path):
    frame = pd.DataFrame([
        {"suite": "synthetic", "task": "mixed30", "model": "nplam", "seed": 10, "mse": 1.0},
        {"suite": "ablations", "task": "mixed30", "model": "nplam", "seed": 10, "mse": 2.0},
    ])
    frame.to_csv(tmp_path / "results.csv", index=False)
    summarize(tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv").set_index("suite")
    assert summary.loc["synthetic", "mse_mean"] == 1.0
    assert summary.loc["ablations", "mse_mean"] == 2.0


def test_offline_cli_full_artifacts(tmp_path):
    config={"seeds":[0],"tasks":[{"name":"mixed30","samples":100}],"hidden":5,"epochs":3,
            "batch_size":32,"patience":3,"fs_threshold":.05,"sd_threshold":.05,"plots":False,
            "models":[{"name":"nplam","grid":[{"lr":.01,"lambdas":[.001,.001,.0001]}]},
                      {"name":"lasso","grid":[{"alpha":.01}]}]}
    p=tmp_path/"config.json"; p.write_text(json.dumps(config))
    out=tmp_path/"run"
    command=[sys.executable,"-m","nplam.cli","run","--config",str(p),"--output",str(out)]
    subprocess.run(command,check=True)
    assert json.loads((out/"manifest.json").read_text())["runs_completed"]==2
    for f in ["results.csv","summary.csv","REPORT.md","mixed30-seed0/preprocessing.json","mixed30-seed0/nplam/model.pt"]:
        assert (out/f).exists()
    assert subprocess.run(command,capture_output=True).returncode!=0
