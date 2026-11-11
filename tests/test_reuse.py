import json
import numpy as np
import pytest
from nplam.experiment import compatible_reuse, environment, reusable_result


def test_reuse_rejects_modified_method_and_settings(tmp_path):
    config={"tasks":[{"name":"mixed30"}]}
    env=environment()
    p=tmp_path/"manifest.json"
    p.write_text(json.dumps({"config":config,"environment":env}))
    compatible_reuse(config,tmp_path)
    with pytest.raises(ValueError,match="configuration"):
        compatible_reuse({"tasks":[{"name":"linear30"}]},tmp_path)
    env["source_hashes"]["src/nplam/model.py"]="changed"
    p.write_text(json.dumps({"config":config,"environment":env}))
    with pytest.raises(ValueError,match="method changed"):
        compatible_reuse(config,tmp_path)


def test_partial_artifact_is_not_reused(tmp_path):
    source=tmp_path/"mixed30-seed10"
    folder=source/"lasso"
    folder.mkdir(parents=True)
    (source/"preprocessing.json").write_text(json.dumps({"fingerprint":"data"}))
    for name in ["selection.json","history.json","model.json"]:
        (folder/name).write_text("{}")
    row={"task":"mixed30","seed":10,"model":"lasso","mse":0.0}
    (folder/"metrics.json").write_text(json.dumps(row))
    target=tmp_path/"copied"
    assert reusable_result(tmp_path,target,"mixed30",10,"lasso","data") is None
    np.savez_compressed(folder/"predictions.npz",y=[1.,2.],pred=[1.,2.])
    assert reusable_result(tmp_path,target,"mixed30",10,"lasso","data")==row
    with pytest.raises(ValueError,match="fingerprint"):
        reusable_result(tmp_path,tmp_path/"other","mixed30",10,"lasso","wrong")
