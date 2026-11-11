param([string]$Python='python', [string]$Root='runs/reproduction')
$ErrorActionPreference='Stop'
foreach ($Name in @('synthetic','highdim','ablations','real')) {
    & $Python -m nplam.cli run --config "configs/$Name.yaml" --output "$Root/$Name"
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $Name" }
}
