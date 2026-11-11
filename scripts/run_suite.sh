#!/usr/bin/env bash
set -euo pipefail
root="${1:-runs/reproduction}"
for name in synthetic highdim ablations real; do
  python -m nplam.cli run --config "configs/$name.yaml" --output "$root/$name"
done
