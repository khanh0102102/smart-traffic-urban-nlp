param(
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Chưa có .venv. Hãy chạy .\scripts\setup.ps1 trước."
}
if ($Reload) {
    & $python -m uvicorn stnlp.api.main:app --host 127.0.0.1 --port 8000 --reload
} else {
    & $python -m uvicorn stnlp.api.main:app --host 127.0.0.1 --port 8000
}
