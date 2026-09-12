# Windows Task Scheduler wrapper: sets cwd so load_dotenv() finds .env, and logs to
# file since the scheduled run has no visible console.
#
# Do NOT go back to `& $python $script 2>&1 | Out-String` with $ErrorActionPreference
# = "Stop". In PowerShell 5.1 each stderr line from a native exe becomes an ErrorRecord,
# so the first line of a Python traceback throws before the pipeline finishes: $output
# is never assigned and the catch block logs only "Traceback (most recent call last):".
# Python writes every traceback to stderr, so that shape truncates the actual exception
# out of the log for all failures - the 2026-09-10 and 2026-09-11 entries lost their
# cause that way. Start-Process with separate redirect files keeps both streams whole.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "etl\daily_ingest.py"
$log = Join-Path $root "logs\daily_ingest.log"

$stdoutFile = [System.IO.Path]::GetTempFileName()
$stderrFile = [System.IO.Path]::GetTempFileName()

"$(Get-Date -Format o) - starting daily_ingest" | Out-File -FilePath $log -Append -Encoding utf8
try {
    $proc = Start-Process -FilePath $python -ArgumentList $script `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile

    $stdout = Get-Content -LiteralPath $stdoutFile -Raw -Encoding UTF8
    $stderr = Get-Content -LiteralPath $stderrFile -Raw -Encoding UTF8

    if ($stdout) { $stdout.TrimEnd() | Out-File -FilePath $log -Append -Encoding utf8 }

    if ($proc.ExitCode -ne 0) {
        # The whole traceback, not just its first line.
        "$(Get-Date -Format o) - FAILED (exit $($proc.ExitCode)):" | Out-File -FilePath $log -Append -Encoding utf8
        if ($stderr) { $stderr.TrimEnd() | Out-File -FilePath $log -Append -Encoding utf8 }
        exit $proc.ExitCode
    }

    # Python can exit 0 and still have written warnings to stderr - keep them.
    if ($stderr) {
        "$(Get-Date -Format o) - stderr (exit 0):" | Out-File -FilePath $log -Append -Encoding utf8
        $stderr.TrimEnd() | Out-File -FilePath $log -Append -Encoding utf8
    }
    "$(Get-Date -Format o) - finished daily_ingest (exit 0)" | Out-File -FilePath $log -Append -Encoding utf8
} catch {
    "$(Get-Date -Format o) - WRAPPER FAILED: $($_ | Out-String)" | Out-File -FilePath $log -Append -Encoding utf8
    throw
} finally {
    Remove-Item -LiteralPath $stdoutFile, $stderrFile -ErrorAction SilentlyContinue
}
