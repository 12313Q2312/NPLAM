"""Public source downloads. No third-party dataset is redistributed by this repo."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
import numpy as np
import pandas as pd

SOURCES = {
    "california": {"url":"https://ndownloader.figshare.com/files/5976036", "license":"Original StatLib: non-commercial use; cached locally only", "license_url":"https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html"},
    "superconductivity": {"url":"https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip", "license":"CC BY 4.0 (UCI)", "license_url":"https://archive.ics.uci.edu/dataset/464/superconductivty+data"},
    "beijing": {"url":"https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip", "license":"CC BY 4.0 (UCI)", "license_url":"https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"},
}


def download(name, cache):
    source = SOURCES[name]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (name + ".archive")
    if not target.exists():
        req = urllib.request.Request(source["url"], headers={"User-Agent":"NPLAM-reproduction/0.1 (research)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = r.read(150_000_001)
        if len(payload) > 150_000_000:
            raise ValueError("unexpected archive size")
        temporary = target.with_suffix(".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = target.with_suffix(".json")
    if manifest.exists() and json.loads(manifest.read_text())["sha256"] != digest:
        raise ValueError("cached data hash mismatch")
    manifest.write_text(json.dumps(dict(source, sha256=digest), indent=2), encoding="utf-8")
    return target


def load_real(name, cache="data", allow_download=False):
    if name not in SOURCES:
        raise ValueError("unsupported dataset (Boston intentionally not bundled)")
    target = Path(cache) / (name+".archive")
    if allow_download:
        target = download(name, cache)
    if not target.exists():
        raise FileNotFoundError(f"{name}: run nplam fetch {name} --cache {cache}")
    manifest = target.with_suffix(".json")
    if not manifest.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != json.loads(manifest.read_text())["sha256"]:
        raise ValueError("missing or invalid download manifest")
    if name == "california":
        with tarfile.open(target) as archive:
            members = [m for m in archive.getmembers() if m.name.endswith("cal_housing.data")]
            if len(members) != 1:
                raise ValueError("California archive layout changed")
            arr = np.loadtxt(archive.extractfile(members[0]), delimiter=",")
        # Original eight predictors: longitude, latitude, age, total rooms,
        # total bedrooms, population, households, median income. No derived ratios.
        return arr[:, :8], arr[:, 8], ["longitude","latitude","age","rooms","bedrooms","population","households","income"]
    with zipfile.ZipFile(target) as archive:
        if name == "superconductivity":
            files = [n for n in archive.namelist() if n.endswith("train.csv")]
            if len(files) != 1:
                raise ValueError("superconductivity archive layout changed")
            df = pd.read_csv(archive.open(files[0]))
            y = df.pop("critical_temp").to_numpy()
            return df.to_numpy(), y, list(df.columns)
        # The UCI outer archive also contains unrelated data.csv/test.csv.
        # Only station files with the published PRSA schema are admissible.
        names = sorted(n for n in archive.namelist() if Path(n).name.startswith("PRSA_Data_") and n.endswith(".csv"))
        if not names:
            nested = [n for n in archive.namelist() if n.endswith(".zip")]
            if len(nested) != 1:
                raise ValueError("Beijing archive layout changed")
            inner = zipfile.ZipFile(io.BytesIO(archive.read(nested[0])))
            frames = [pd.read_csv(inner.open(n)) for n in sorted(inner.namelist()) if Path(n).name.startswith("PRSA_Data_") and n.endswith(".csv")]
        else:
            frames = [pd.read_csv(archive.open(n)) for n in names]
    df = pd.concat(frames, ignore_index=True).sort_values(["year","month","day","hour","station"], kind="stable")
    # Explicit deterministic ordinal encoding; arbitrary direction order is a limitation.
    compass = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    df["wd"] = df["wd"].map({k:i for i,k in enumerate(compass)})
    cols = ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","wd","WSPM"]
    df = df.dropna(subset=cols+["PM2.5"])
    return df[cols].to_numpy(), df["PM2.5"].to_numpy(), cols
