<#
Keeps a Spot training VM alive from your own machine: every few minutes it checks the VM and

  * RUNNING (or starting/stopping)  -> nothing to do;
  * stopped because it was PREEMPTED -> start it again; if the zone has no capacity for
    -StartTries checks in a row, snapshot the boot disk and create the VM in the first zone of
    -Zones that has capacity, then delete the old VM and the snapshot once the new one is up;
  * stopped any other way (run.sh --shutdown at the end of the run, or you pressed Stop)
    -> log it and exit.

The VM resumes training by itself at boot (cloud/startup-script.sh + ~/autoresume.sh,
see cloud/GUIDE.zh.md step 7), so the watchdog only has to get a VM running.
Your machine must stay on and awake while it runs. Log: outputs\watchdog.log.

  powershell -NoProfile -ExecutionPolicy Bypass -File cloud\watchdog.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File cloud\watchdog.ps1 -Once   # one check, no action
#>
param(
  [string]$Instance = "simpo-l4",
  [string]$Zone = "us-central1-a",
  [string[]]$Zones = @("us-central1-a", "us-central1-b", "us-central1-c", "us-east1-b", "us-east1-c",
                       "us-east1-d", "us-east4-a", "us-east4-c", "us-west1-a", "us-west1-b", "us-west1-c",
                       "us-west4-a", "us-west4-c"),
  [string]$User = "benzeng03",
  [string]$MachineType = "g2-standard-8",
  [int]$IntervalMinutes = 5,
  [int]$StartTries = 6,
  [string]$StartupScript = "$PSScriptRoot\startup-script.sh",
  [string]$Log = "$PSScriptRoot\..\outputs\watchdog.log",
  [switch]$Once
)

$gcloud = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (-not (Test-Path $gcloud)) { $gcloud = (Get-Command gcloud -ErrorAction Stop).Source }

function Write-Log([string]$msg) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

function Invoke-Gcloud {
  # returns @{ok; out}; stderr folded into the text, never thrown
  $out = & $gcloud @args 2>&1 | ForEach-Object { "$_" }
  return @{ ok = ($LASTEXITCODE -eq 0); out = ($out -join "`n") }
}

function Get-Status([string]$zone) {
  $r = Invoke-Gcloud compute instances describe $Instance --zone $zone --format "value(status)"
  if ($r.ok) { return $r.out.Trim() } else { return "MISSING" }
}

function Get-LastEvent([string]$zone) {
  # newest successful lifecycle operation on this instance. Skipped: setMetadata etc. (gcloud ssh
  # adds keys that way) and failed operations (a start refused for lack of capacity), either of
  # which would otherwise hide the preemption
  $lifecycle = @("compute.instances.preempted", "compute.instances.guestTerminate", "stop", "start",
                 "insert", "reset", "suspend", "resume", "delete")
  $r = Invoke-Gcloud compute operations list --zones $zone --sort-by "~insertTime" --limit 500 `
        --format "csv[no-heading](operationType,insertTime,targetLink,error.errors[0].code)"
  if (-not $r.ok) { return $null }
  foreach ($line in ($r.out -split "`n")) {
    $f = $line.Trim().Split(",")
    if ($f.Count -ge 3 -and $lifecycle -contains $f[0] -and $f[2] -match "/instances/$([regex]::Escape($Instance))$" `
        -and ($f.Count -lt 4 -or $f[3] -eq "")) {
      return @{ type = $f[0]; time = $f[1] }
    }
  }
  return $null
}

function Test-Resumed([string]$zone) {
  $r = Invoke-Gcloud compute ssh "$User@$Instance" --zone $zone --quiet --command "tmux has-session -t simpo && echo SIMPO-OK"
  return ($r.out -match "SIMPO-OK")
}

function Move-Instance {
  $disk = (Invoke-Gcloud compute instances describe $Instance --zone $script:Zone --format "value(disks[0].source.basename())").out.Trim()
  $snap = "$Instance-move-" + (Get-Date -Format "yyyyMMdd-HHmmss")
  Write-Log "snapshotting disk $disk ($($script:Zone)) as $snap"
  $r = Invoke-Gcloud compute snapshots create $snap --source-disk $disk --source-disk-zone $script:Zone
  if (-not $r.ok) { Write-Log "snapshot failed: $($r.out)"; return $false }
  foreach ($z in $Zones) {
    if ($z -eq $script:Zone) { continue }
    $r = Invoke-Gcloud compute instances create $Instance --zone $z --machine-type $MachineType `
          --accelerator "count=1,type=nvidia-l4" --provisioning-model SPOT --instance-termination-action STOP `
          --maintenance-policy TERMINATE --no-restart-on-failure --scopes default `
          --metadata-from-file "startup-script=$StartupScript" `
          --create-disk "boot=yes,auto-delete=yes,source-snapshot=$snap,size=100,type=pd-balanced,name=$Instance-$z"
    if ($r.ok) {
      Write-Log "created $Instance in $z from $snap"
      Start-Sleep -Seconds 240
      if (Test-Resumed $z) {
        Write-Log "training resumed in $z; deleting the old VM in $($script:Zone) and snapshot $snap"
        Invoke-Gcloud compute instances delete $Instance --zone $script:Zone --quiet | Out-Null
        Invoke-Gcloud compute snapshots delete $snap --quiet | Out-Null
      } else {
        Write-Log "WARNING: new VM in $z is up but tmux session 'simpo' not found; old VM and $snap kept"
      }
      $script:Zone = $z
      return $true
    }
    $why = if ($r.out -match "RESOURCE_POOL_EXHAUSTED|unavailable|stockout") { "no capacity" } else { ($r.out -split "`n" | Select-Object -Last 2) -join " " }
    Write-Log "  $z : $why"
  }
  Write-Log "no zone had capacity; deleting snapshot $snap, will retry"
  Invoke-Gcloud compute snapshots delete $snap --quiet | Out-Null
  return $false
}

New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
if (-not $Once) {
  # keep Windows from idle-sleeping while the watchdog runs (released when it exits;
  # a closed lid or choosing Sleep still sleeps)
  Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
  [Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
}
Write-Log "watchdog started: $Instance in $Zone, every $IntervalMinutes min"
$failedStarts = 0
$missing = 0
while ($true) {
  $status = Get-Status $Zone
  if ($status -ne "MISSING") { $missing = 0 }
  if ($status -eq "TERMINATED") {
    $ev = Get-LastEvent $Zone
    $type = if ($ev) { $ev.type } else { "unknown" }
    if ($Once) { Write-Log "status TERMINATED, last event $type"; break }
    if ($type -ne "compute.instances.preempted") {
      Write-Log "VM stopped by '$type' (finished run or manual stop), not preempted: watchdog exits"
      break
    }
    $r = Invoke-Gcloud compute instances start $Instance --zone $Zone
    if ($r.ok) {
      Write-Log "preempted at $($ev.time); restarted in $Zone (training resumes at boot)"
      $failedStarts = 0
    } else {
      $failedStarts++
      Write-Log "preempted; start failed in $Zone ($failedStarts/$StartTries): $(if ($r.out -match 'RESOURCE_POOL_EXHAUSTED|unavailable|stockout') { 'no capacity' } else { $r.out })"
      if ($failedStarts -ge $StartTries) {
        if (Move-Instance) { $failedStarts = 0 }
      }
    }
  } elseif ($status -eq "MISSING") {
    # a single failed describe call is often transient (network, token refresh): exit only after
    # three checks in a row find no instance
    $missing++
    if ($missing -ge 3 -or $Once) {
      Write-Log "instance $Instance not found in ${Zone} ($missing checks in a row): watchdog exits"
      break
    }
    Write-Log "instance $Instance not found in ${Zone} ($missing/3); checking again"
  } elseif ($Once) {
    $ev = Get-LastEvent $Zone
    Write-Log "status $status, last lifecycle event $(if ($ev) { "$($ev.type) at $($ev.time)" } else { 'none' })"
    break
  }
  Start-Sleep -Seconds ($IntervalMinutes * 60)
}
