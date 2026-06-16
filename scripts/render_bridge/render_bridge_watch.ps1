# render_bridge_watch.ps1 — pair with sync_runs_to_windows.sh (WSL side).
#
# Polls a directory for new run sub-directories. When the newest sub-directory
# changes AND contains at least one .rrd file, kills any running rerun process
# and re-launches it on every .rrd in that new run.
#
# Usage (from PowerShell, on Windows):
#   .\render_bridge_watch.ps1 [-RunsDir <path>] [-PollSeconds 3] [-RerunPath rerun]
#
# Defaults:
#   RunsDir     = %USERPROFILE%\leapsim_runs  (matches the sh script default)
#   PollSeconds = 3
#   RerunPath   = "rerun"  (must be on PATH; install via `pip install rerun-sdk`)
#
# Notes:
# - Detection is "newest dir by LastWriteTime that contains *.rrd". So an empty
#   dir created mid-sync won't trigger a launch; we wait for files to arrive.
# - We pass every .rrd in the run dir to a single rerun invocation, which loads
#   them as separate recordings the viewer can switch between.
# - Press Ctrl+C to stop.

param(
    [string]$RunsDir    = "$env:USERPROFILE\leapsim_runs",
    [int]   $PollSeconds = 3,
    [string]$RerunPath  = "rerun"
)

if (-not (Test-Path $RunsDir)) {
    Write-Host "[watch] runs dir does not exist yet: $RunsDir"
    Write-Host "[watch] waiting for it to appear..."
    while (-not (Test-Path $RunsDir)) { Start-Sleep -Seconds $PollSeconds }
}

Write-Host "[watch] watching $RunsDir"
Write-Host "[watch] poll every $PollSeconds s, launching with: $RerunPath"

$currentRun = ""

while ($true) {
    try {
        $latest = Get-ChildItem -Path $RunsDir -Directory -ErrorAction Stop |
                  Sort-Object LastWriteTime -Descending |
                  Select-Object -First 1

        if ($latest -and $latest.FullName -ne $currentRun) {
            $rrdFiles = Get-ChildItem -Path $latest.FullName -Filter "*.rrd" `
                        -Recurse -ErrorAction SilentlyContinue
            if ($rrdFiles -and $rrdFiles.Count -gt 0) {
                $ts = Get-Date -Format 'HH:mm:ss'
                Write-Host "[watch] $ts  switching to: $($latest.Name) ($($rrdFiles.Count) .rrd files)"

                # Kill any running rerun. -ErrorAction SilentlyContinue so first
                # iteration (nothing to kill) doesn't error.
                Get-Process rerun -ErrorAction SilentlyContinue | Stop-Process -Force
                Start-Sleep -Milliseconds 500

                # Launch rerun with every .rrd. Quote each path; Start-Process
                # joins -ArgumentList with spaces.
                $args = $rrdFiles | ForEach-Object { '"' + $_.FullName + '"' }
                Start-Process -FilePath $RerunPath -ArgumentList $args

                $currentRun = $latest.FullName
            }
        }
    } catch {
        Write-Warning "[watch] poll failed: $_"
    }
    Start-Sleep -Seconds $PollSeconds
}
