param(
    [switch]$SkipDevDependencies
)

$ErrorActionPreference = "Stop"

Write-Host "== ST-NLP environment setup ==" -ForegroundColor Cyan
$python = Get-Command py -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Không tìm thấy Python Launcher (py). Hãy cài Python 3.11 64-bit trước."
}

py -3.11 --version
if ($LASTEXITCODE -ne 0) {
    throw "Không tìm thấy Python 3.11. Hãy cài Python 3.11 64-bit."
}

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

$venvPython = ".\.venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
if (-not $SkipDevDependencies) {
    & $venvPython -m pip install -r requirements-dev.txt
}
& $venvPython -m pip install -e .

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Đã tạo .env từ .env.example. Hãy đổi API_ADMIN_KEY trước khi sử dụng."
}

Write-Host "Setup hoàn tất. Kích hoạt bằng .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
