<#
.SYNOPSIS
Train fixed and adaptive SimPO variants, select the fixed baseline on dev pairs,
then generate AlpacaEval answers and judge them with PairRM.

.EXAMPLE
.\run_experiment.ps1 -WhatIf
.\run_experiment.ps1
.\run_experiment.ps1 -Variants "fixed:0.5","erf-mean:2" -SkipEval
.\run_experiment.ps1 -RunDir outputs\run-20261120-0100 -EvalOnly

Variant specs:  fixed:<gamma> | erf-zero:<gamma_max> | erf-mean:<gamma_max>
                | tanh-zero:<gamma_max> | tanh-mean:<gamma_max> | amapo
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string[]]$Variants = @("fixed:0", "fixed:0.25", "fixed:0.5", "fixed:1",
                            "erf-zero:2", "erf-mean:2", "tanh-mean:2", "amapo"),
    [string]$RunDir,
    [string]$Python = (Join-Path $PSScriptRoot "..\RobustSimPO\.venv\Scripts\python.exe"),
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

function Convert-Variant([string]$Spec) {
    $kind, $value = $Spec.Split(":", 2)
    if ($kind -eq "amapo") { return [pscustomobject]@{ Name = "amapo"; Args = @("--margin", "amapo") } }
    if ($null -eq $value) { throw "Variant '$Spec' needs a value, e.g. fixed:0.5" }
    $v = [double]::Parse($value, $inv).ToString("R", $inv)
    if ($kind -eq "fixed") { return [pscustomobject]@{ Name = "fixed-g$v"; Args = @("--margin", "fixed", "--gamma", $v) } }
    if ($kind -match '^(erf|tanh)-(zero|mean)$') {
        return [pscustomobject]@{ Name = "$kind-g$v"; Args = @("--margin", "adaptive", "--mapping", $Matches[1],
                                              "--threshold", $Matches[2], "--gamma-max", $v) }
    }
    throw "Unknown variant '$Spec'"
}

if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python interpreter: $Python (pass -Python)" }
if (!$RunDir) {
    if ($EvalOnly) { throw "-EvalOnly needs -RunDir" }
    $RunDir = Join-Path $PSScriptRoot "outputs\run-bs$BatchSize-seed$Seed-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
$plan = $Variants | ForEach-Object { Convert-Variant $_ }
Write-Host "Run directory: $RunDir"
Write-Host "Variants: $(($plan | ForEach-Object { $_.Name }) -join ', ')"

Push-Location -LiteralPath $PSScriptRoot
try {
    if (!$EvalOnly) {
        if (!$PSCmdlet.ShouldProcess($RunDir, "Train $($plan.Count) adapters")) { return }
        & $Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
        Assert-Exit
        New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
        $common = @("--batch-size", "$BatchSize", "--seed", "$Seed", "--lr", $LearningRate.ToString("R", $inv),
                    "--beta", $Beta.ToString("R", $inv), "--epochs", "1", "--max-length", "384", "--max-prompt", "192")
        foreach ($item in $plan) {
            $out = Join-Path $RunDir $item.Name
            if (Test-Path -LiteralPath (Join-Path $out "margin_stats.json")) { Write-Host "Skip trained $($item.Name)"; continue }
            if (Test-Path -LiteralPath $out) { Remove-Item -Recurse -Force -LiteralPath $out }
            Write-Host "=== Training $($item.Name)"
            & $Python -u train.py @common @($item.Args) --output $out
            Assert-Exit
        }
        & $Python -u score_pairs.py $RunDir
        Assert-Exit
    }
    if ($SkipEval) { return }

    $selection = Get-Content -Raw -LiteralPath (Join-Path $RunDir "selection.json") | ConvertFrom-Json
    $baseline = $selection.best_fixed
    if (!$baseline) { throw "No fixed-margin adapter to use as baseline" }
    $adapterArgs = @()
    foreach ($dir in Get-ChildItem -LiteralPath $RunDir -Directory) {
        if (!(Test-Path -LiteralPath (Join-Path $dir.FullName "adapter_model.safetensors"))) { continue }
        # Generation compares the selected fixed baseline, the mean-matched fixed margin and all adaptive rules.
        if ($dir.Name -like "fixed-g*" -and $dir.Name -ne $baseline -and $dir.Name -ne "fixed-g0.5") { continue }
        $adapterArgs += @("--adapter", "$($dir.Name)=$($dir.FullName)")
    }
    $gen = Join-Path $RunDir "generations-ae$NumPrompts-seed$Seed"
    if (!$PSCmdlet.ShouldProcess($gen, "Generate and judge with baseline $baseline")) { return }
    Write-Host "Baseline (dev-selected fixed): $baseline"
    $baseArg = @(if ($IncludeBase) { "--base" })
    & $Python -u generate.py @adapterArgs @baseArg --output $gen --num-prompts $NumPrompts --seed $Seed --max-new-tokens $MaxNewTokens
    Assert-Exit
    & $Python -u judge.py $gen --baseline $baseline
    Assert-Exit
    & $Python -u summarize.py $RunDir
    Assert-Exit
} finally {
    Pop-Location
}
