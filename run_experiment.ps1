<#
.SYNOPSIS
Train fixed and adaptive SimPO variants, select the fixed baseline on dev pairs,
then generate AlpacaEval answers and judge them with PairRM.

.EXAMPLE
.\run_experiment.ps1 -WhatIf
.\run_experiment.ps1
.\run_experiment.ps1 -Variants "fixed:0.5","erf-zero:2" -SkipEval
.\run_experiment.ps1 -RunDir outputs\run-bs8-seed42-20261120-010000 -EvalOnly

Variant specs:  fixed:<gamma> | erf-zero:<gamma_max>
Re-running with an existing -RunDir trains only missing variants and reuses cached generations.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string[]]$Variants = @("fixed:0", "fixed:0.25", "fixed:0.5", "fixed:0.75", "fixed:1", "erf-zero:2"),
    [string]$RunDir,
    [string]$Python,
    [int]$BatchSize = 8,
    [int]$Seed = 42,
    [double]$LearningRate = 0.00005,
    [double]$Beta = 2,
    [int]$NumPrompts = 64,
    [int]$MaxNewTokens = 4096,
    [switch]$IncludeBase,
    [switch]$SkipEval,
    [switch]$EvalOnly
)
$ErrorActionPreference = "Stop"
$inv = [cultureinfo]::InvariantCulture

function Assert-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE)." } }

# Run Python and fail only on a nonzero exit code. Stderr (warnings, progress bars) is
# forwarded as plain text so that PowerShell 5.1 never turns it into a terminating error.
function Invoke-Python {
    $ErrorActionPreference = "Continue"
    & $Python @args 2>&1 | ForEach-Object { Write-Host "$_" }
    Assert-Exit
}

function Convert-Variant([string]$Spec) {
    $kind, $value = $Spec.Split(":", 2)
    if ($null -eq $value) { throw "Variant '$Spec' needs a value, e.g. fixed:0.5" }
    $v = [double]::Parse($value, $inv).ToString("R", $inv)
    if ($kind -eq "fixed") { return [pscustomobject]@{ Name = "fixed-g$v"; Args = @("--margin", "fixed", "--gamma", $v) } }
    if ($kind -eq "erf-zero") { return [pscustomobject]@{ Name = "erf-zero-g$v"; Args = @("--margin", "adaptive", "--gamma-max", $v) } }
    throw "Unknown variant '$Spec' (use fixed:<gamma> or erf-zero:<gamma_max>)"
}

# Resolved here because $PSScriptRoot is empty in param defaults under `powershell -File` (PS 5.1).
if (!$Python) { $Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" }
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python interpreter: $Python (pass -Python)" }
if (!$RunDir) {
    if ($EvalOnly) { throw "-EvalOnly needs -RunDir" }
    $RunDir = Join-Path $PSScriptRoot "outputs\run-bs$BatchSize-seed$Seed-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
$plan = @($Variants | ForEach-Object { Convert-Variant $_ })
$names = @($plan | ForEach-Object { $_.Name })
if (!($names | Where-Object { $_ -like "fixed-g*" })) { throw "Need at least one fixed:<gamma> variant as the baseline" }
Write-Host "Run directory: $RunDir"
Write-Host "Variants: $($names -join ', ')"

Push-Location -LiteralPath $PSScriptRoot
try {
    if (!$EvalOnly) {
        if (!$PSCmdlet.ShouldProcess($RunDir, "Train $($plan.Count) adapters")) { return }
        Invoke-Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
        New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
        $common = @("--batch-size", "$BatchSize", "--seed", "$Seed", "--lr", $LearningRate.ToString("R", $inv),
                    "--beta", $Beta.ToString("R", $inv), "--epochs", "1", "--max-length", "384", "--max-prompt", "192")
        foreach ($item in $plan) {
            $out = Join-Path $RunDir $item.Name
            if (Test-Path -LiteralPath (Join-Path $out "margin_stats.json")) { Write-Host "Skip trained $($item.Name)"; continue }
            if (Test-Path -LiteralPath $out) { Remove-Item -Recurse -Force -LiteralPath $out }
            Write-Host "=== Training $($item.Name)"
            Invoke-Python -u train.py @common @($item.Args) --output $out
        }
    }
    if (!$EvalOnly -or !(Test-Path -LiteralPath (Join-Path $RunDir "selection.json"))) {
        Invoke-Python -u score_pairs.py $RunDir --adapters @names
    }
    if ($SkipEval) { return }

    $selection = Get-Content -Raw -LiteralPath (Join-Path $RunDir "selection.json") | ConvertFrom-Json
    $baseline = $selection.best_fixed
    if (!$baseline) { throw "No fixed-margin adapter to use as baseline" }
    # Generation compares the dev-selected fixed baseline with every adaptive variant.
    $adapterArgs = @()
    foreach ($name in $names) {
        if ($name -like "fixed-g*" -and $name -ne $baseline) { continue }
        $dir = Join-Path $RunDir $name
        if (!(Test-Path -LiteralPath (Join-Path $dir "adapter_model.safetensors"))) { throw "Missing adapter: $dir" }
        $adapterArgs += @("--adapter", "$name=$((Resolve-Path -LiteralPath $dir).Path)")
    }
    $gen = Join-Path $RunDir "generations-ae$NumPrompts-seed$Seed"
    if (!$PSCmdlet.ShouldProcess($gen, "Generate and judge with baseline $baseline")) { return }
    Write-Host "Baseline (dev-selected fixed): $baseline"
    $baseArg = @(if ($IncludeBase) { "--base" })
    Invoke-Python -u generate.py @adapterArgs @baseArg --output $gen --num-prompts $NumPrompts --seed $Seed --max-new-tokens $MaxNewTokens
    Invoke-Python -u judge.py $gen --baseline $baseline
    Invoke-Python -u summarize.py $RunDir
} finally {
    Pop-Location
}
