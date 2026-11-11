import hashlib
import io
import json
import zipfile
import numpy as np
from nplam.real_data import load_real


def test_beijing_uses_official_station_files_not_outer_csv(tmp_path):
    header="No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,wd,WSPM,station\n"
    rows="".join(f"{i},2013,3,1,{i},2,3,4,5,6,7,8,1000,3,0,N,2,A\n" for i in range(12))
    inner=io.BytesIO()
    with zipfile.ZipFile(inner,"w") as z:
        z.writestr("PRSA_Data_A_20130301-20170228.csv",header+rows)
    p=tmp_path/"beijing.archive"
    with zipfile.ZipFile(p,"w") as z:
        z.writestr("data.csv","unrelated,data\n1,2")
        z.writestr("test.csv","wrong,schema\n1,2")
        z.writestr("PRSA2017_Data_20130301-20170228.zip",inner.getvalue())
    p.with_suffix(".json").write_text(json.dumps({"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}))
    x,y,names=load_real("beijing",tmp_path)
    assert x.shape==(12,11) and len(names)==11
    np.testing.assert_array_equal(y,np.full(12,2))
