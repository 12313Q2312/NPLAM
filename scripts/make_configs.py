"""Generate explicit budget presets. This script does not run experiments."""
from copy import deepcopy
from pathlib import Path
import yaml

base=yaml.safe_load(Path("configs/pilot.yaml").read_text())
base["models"] += [
    {"name":"mlp","grid":[{"lr":.003},{"lr":.01}]},
    {"name":"nam","grid":[{"lr":.003},{"lr":.01}]},
    {"name":"lasso","grid":[{"alpha":a} for a in [.0001,.001,.01,.1]]}]
synthetic=deepcopy(base)
synthetic["seeds"]=list(range(10,20))
synthetic["tasks"]=[{"name":n,"samples":1600} for n in ["mixed30","linear30","nonlinear30","dense30"]]
high=deepcopy(base)
high.update(seeds=[10,11,12],epochs=250,patience=70,tasks=[{"name":"sparse300","samples":2000}])
real=deepcopy(base)
real.update(seeds=[10,11,12],epochs=200,patience=60,tasks=[{"name":n,"samples":4096} for n in ["california","superconductivity","beijing"]])
abl=deepcopy(base)
abl.update(seeds=[10,11,12],tasks=[{"name":"mixed30","samples":1600}])
abl["models"]=[deepcopy(base["models"][0])]
abl["models"][0]["grid"]=[base["models"][0]["grid"][0]]
for name in ["no_l1","no_l2","no_fs","no_sd","no_reg","train_hidden"]:
    spec=deepcopy(abl["models"][0]); spec.update(label=name,ablation=name)
    abl["models"].append(spec)
legacy=deepcopy(abl["models"][0])
legacy.update(label="legacy",mode="legacy",grid=[{"lr":.01,"lambdas":[.01,.01,.0005]}])
abl["models"].append(legacy)
for name,obj in [("synthetic",synthetic),("highdim",high),("real",real),("ablations",abl)]:
    Path(f"configs/{name}.yaml").write_text(yaml.safe_dump(obj,sort_keys=False),encoding="utf-8")
# Large preset is executable, not evidence that it was run. Full coarse-to-fine
# search and original baselines still need independent work; name stays explicit.
extended=deepcopy(high)
extended.update(seeds=list(range(10)),epochs=800,batch_size=100,patience=800)
extended["tasks"][0]["samples"]=16000
Path("configs/extended_300d.yaml").write_text(yaml.safe_dump(extended,sort_keys=False),encoding="utf-8")
