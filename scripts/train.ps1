param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,
    [int]$Epochs = 12
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Dataset)) {
    throw "Không tìm thấy dataset: $Dataset"
}
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Chưa có .venv. Hãy chạy .\scripts\setup.ps1 trước."
}
& $python -m stnlp.training.train --data $Dataset --artifacts-dir artifacts --epochs $Epochs
if ($LASTEXITCODE -ne 0) {
    throw "Huấn luyện thất bại. Kiểm tra log phía trên."
}
