<#
.SYNOPSIS
Train fixed-margin and calibrated-margin SimPO variants, select the fixed baseline on dev pairs,
then generate AlpacaEval answers and judge them with a GPT model (research/llm_judge.py, Codex CLI).

Dev pairs select the fixed baseline; test pairs are scored for reporting only.

.EXAMPLE
.\run_experiment.ps1 -WhatIf
.\run_experiment.ps1
.\run_experiment.ps1 -Variants "fixed:2","calibrated" -SkipEval
.\run_experiment.ps1 -RunDir outputs\run-bs16-lr0.0001-ep1-seed42-C -EvalOnly -Variants "fixed:0","fixed:1","fixed:2","calibrated"

Variant specs:  fixed:<gamma> | calibrated[-logistic][-ema:<decay>]
  calibrated: gamma_i = (1 - k) r_i with k = 2 mean / var of the current batch's reward gaps
  calibrated-ema:0.9: the same with k from moving averages of the gap moments (train.py --k-ema 0.9)
  calibrated-logistic: logistic location reference instead of the normal, a saturating logit (--k-family logistic)
With -SkipEval no fixed variant is required (no baseline is needed). With -EvalOnly pass the same
-Variants as the run directory contains: the adapter list is built from them.
Re-running with an existing -RunDir trains only missing variants and reuses cached generations.
-GenerateVariants limits generation to the listed non-fixed variants; the dev-selected fixed
baseline is always generated. -GenerateGroup sets how many prompts are sampled together (and share
one sampling seed); keep it fixed within a generations directory.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string[]]$Variants = @("fixed:0", "fixed:1", "fixed:2", "calibrated"),
    [string]$RunDir,
    [string]$Python,
    [string]$TrainData,
    [string]$DevData,
    [string]$TestData,
    [int]$BatchSize = 16,
    [int]$Seed = 42,
    [double]$LearningRate = 0.0001,
    [ValidateRange(1, 100)][int]$Epochs = 1,
    [double]$Beta = 2,
    [int]$NumPrompts = 128,
    [string[]]$GenerateVariants,
    [ValidateRange(1, 256)][int]$GenerateGroup = 16,
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
    if ($Spec -match '^calibrated-logistic(-ema:(.+))?$') {
        $a = @("--margin", "calibrated", "--k-family", "logistic")
        if (!$Matches[2]) { return [pscustomobject]@{ Name = "calibrated-logistic"; Args = $a } }
        $b = [double]::Parse($Matches[2], $inv).ToString("R", $inv)
        return [pscustomobject]@{ Name = "calibrated-logistic-ema$b"; Args = $a + @("--k-ema", $b) }
    }
    if ($Spec -match '^calibrated-ema:(.+)$') {
        $b = [double]::Parse($Matches[1], $inv).ToString("R", $inv)
        return [pscustomobject]@{ Name = "calibrated-ema$b"; Args = @("--margin", "calibrated", "--k-ema", $b) }
    }
    if ($Spec -match '^calibrated(:.*)?$') {
        return [pscustomobject]@{ Name = "calibrated"; Args = @("--margin", "calibrated") }
    }
    $kind, $value = $Spec.Split(":", 2)
    if ($kind -eq "fixed" -and $null -ne $value) {
        $v = [double]::Parse($value, $inv).ToString("R", $inv)
        return [pscustomobject]@{ Name = "fixed-g$v"; Args = @("--margin", "fixed", "--gamma", $v) }
    }
    throw "Unknown variant '$Spec' (use fixed:<gamma> or calibrated[-logistic][-ema:<decay>])"
}

# Resolved here because $PSScriptRoot is empty in param defaults under `powershell -File` (PS 5.1).
if (!$Python) { $Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" }
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python interpreter: $Python (pass -Python)" }
if (!$TrainData) { $TrainData = Join-Path $PSScriptRoot "data\uf_train4096.jsonl" }
if (!$DevData) { $DevData = Join-Path $PSScriptRoot "data\uf_dev500.jsonl" }
if (!$TestData) { $TestData = Join-Path $PSScriptRoot "data\uf_test1486.jsonl" }
foreach ($f in @($TrainData, $DevData, $TestData)) {
    if (!(Test-Path -LiteralPath $f -PathType Leaf)) { throw "Missing data file: $f (run prepare_data.py)" }
}
if (!$RunDir) {
    if ($EvalOnly) { throw "-EvalOnly needs -RunDir" }
    $lrTag = $LearningRate.ToString("R", $inv)
    $RunDir = Join-Path $PSScriptRoot "outputs\run-bs$BatchSize-lr$lrTag-ep$Epochs-seed$Seed-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
$plan = @($Variants | ForEach-Object { Convert-Variant $_ })
$names = @($plan | ForEach-Object { $_.Name })
if (!$SkipEval -and !($names | Where-Object { $_ -like "fixed-g*" })) { throw "Need at least one fixed:<gamma> variant as the baseline" }
Write-Host "Run directory: $RunDir"
Write-Host "Variants: $($names -join ', ')"

Push-Location -LiteralPath $PSScriptRoot
try {
    if (!$EvalOnly) {
        if (!$PSCmdlet.ShouldProcess($RunDir, "Train $($plan.Count) adapters")) { return }
        Invoke-Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
        New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
        $common = @("--data", $TrainData, "--batch-size", "$BatchSize", "--seed", "$Seed", "--lr", $LearningRate.ToString("R", $inv),
                    "--beta", $Beta.ToString("R", $inv), "--epochs", "$Epochs", "--max-length", "384", "--max-prompt", "192")
        foreach ($item in $plan) {
            $out = Join-Path $RunDir $item.Name
            if (Test-Path -LiteralPath (Join-Path $out "margin_stats.json")) { Write-Host "Skip trained $($item.Name)"; continue }
            if (Test-Path -LiteralPath $out) { Remove-Item -Recurse -Force -LiteralPath $out }
            Write-Host "=== Training $($item.Name)"
            Invoke-Python -u train.py @common @($item.Args) --output $out
        }
    }
    if (!$EvalOnly -or !(Test-Path -LiteralPath (Join-Path $RunDir "selection.json"))) {
        Invoke-Python -u score_pairs.py $RunDir --split dev --data $DevData --adapters @names
    }
    if (!$EvalOnly -or !(Test-Path -LiteralPath (Join-Path $RunDir "test_scores.json"))) {
        Invoke-Python -u score_pairs.py $RunDir --split test --data $TestData --adapters @names
    }
    Invoke-Python -u summarize.py $RunDir
    if ($SkipEval) { return }

    $selection = Get-Content -Raw -LiteralPath (Join-Path $RunDir "selection.json") | ConvertFrom-Json
    $baseline = $selection.best_fixed
    if (!$baseline) { throw "No fixed-margin adapter to use as baseline" }
    # Generation compares the dev-selected fixed baseline with every non-fixed variant
    # (or only those in -GenerateVariants).
    $genNames = if ($GenerateVariants) { @($GenerateVariants | ForEach-Object { (Convert-Variant $_).Name }) } else { $null }
    $adapterArgs = @()
    foreach ($name in $names) {
        if ($name -like "fixed-g*" -and $name -ne $baseline) { continue }
        if ($genNames -and $name -ne $baseline -and $genNames -notcontains $name) { continue }
        $dir = Join-Path $RunDir $name
        if (!(Test-Path -LiteralPath (Join-Path $dir "adapter_model.safetensors"))) { throw "Missing adapter: $dir" }
        $adapterArgs += @("--adapter", "$name=$((Resolve-Path -LiteralPath $dir).Path)")
    }
    $groupTag = if ($GenerateGroup -ne 8) { "-g$GenerateGroup" } else { "" }  # caches depend on the group size
    $gen = Join-Path $RunDir "generations-ae$NumPrompts-seed$Seed$groupTag"
    if (!$PSCmdlet.ShouldProcess($gen, "Generate and judge with baseline $baseline")) { return }
    Write-Host "Baseline (dev-selected fixed): $baseline"
    $baseArg = @(if ($IncludeBase) { "--base" })
    Invoke-Python -u generate.py @adapterArgs @baseArg --train-data $TrainData --output $gen --num-prompts $NumPrompts --seed $Seed --group $GenerateGroup --max-new-tokens $MaxNewTokens
    foreach ($name in $names) {
        if ($name -like "fixed-g*") { continue }
        if ($genNames -and $genNames -notcontains $name) { continue }
        Invoke-Python -u research\llm_judge.py $gen --variant $name --baseline $baseline --model gpt-6.2-sol --effort medium
    }
    Invoke-Python -u summarize.py $RunDir
} finally {
    Pop-Location
}
