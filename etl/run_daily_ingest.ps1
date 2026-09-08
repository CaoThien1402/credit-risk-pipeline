# Buổi 5: wrapper cho Windows Task Scheduler — đảm bảo chạy đúng cwd (để load_dotenv()
# tìm thấy .env) và ghi log lại vì scheduled task không hiện console output.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "etl\daily_ingest.py"
$log = Join-Path $root "logs\daily_ingest.log"

"$(Get-Date -Format o) - starting daily_ingest" | Out-File -FilePath $log -Append -Encoding utf8
try {
    $output = & $python $script 2>&1 | Out-String
    $output | Out-File -FilePath $log -Append -Encoding utf8
    "$(Get-Date -Format o) - finished daily_ingest (exit $LASTEXITCODE)" | Out-File -FilePath $log -Append -Encoding utf8
} catch {
    "$(Get-Date -Format o) - FAILED: $_" | Out-File -FilePath $log -Append -Encoding utf8
    throw
}
