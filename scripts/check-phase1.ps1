$ErrorActionPreference = "Stop"

$requiredDirectories = @(
    "backend", "backend/app", "backend/app/api", "backend/app/api/routes", "backend/app/core",
    "frontend", "frontend/citizen", "frontend/admin", "frontend/assets", "frontend/assets/css", "frontend/assets/js",
    "ml", "ml/data/raw", "ml/data/interim", "ml/data/processed",
    "ml/src/preprocessing", "ml/src/features", "ml/src/models", "ml/src/evaluation",
    "ml/notebooks", "ml/artifacts", "ml/reports",
    "database", "database/migrations", "database/seeds", "docs", "scripts", "tests"
)

$requiredFiles = @(
    ".gitignore", ".gitattributes", ".editorconfig", ".env.example", "README.md", "pyproject.toml",
    "backend/requirements.txt", "backend/requirements-dev.txt", "backend/app/main.py",
    "backend/app/api/routes/health.py", "frontend/citizen/index.html", "frontend/admin/index.html",
    "frontend/assets/css/style.css", "frontend/assets/js/app.js", "scripts/check-phase1.ps1", "tests/test_health.py"
)

$errors = [System.Collections.Generic.List[string]]::new()
$gitRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0 -or -not $gitRoot) {
    $errors.Add("Current directory is not inside a Git repository.")
}

foreach ($path in $requiredDirectories) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { $errors.Add("Missing directory: $path") }
}
foreach ($path in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $errors.Add("Missing file: $path") }
}

git check-ignore --quiet --no-index .env
if ($LASTEXITCODE -ne 0) { $errors.Add(".env is not ignored by Git.") }

git check-ignore --quiet --no-index .env.example
if ($LASTEXITCODE -eq 0) { $errors.Add(".env.example is ignored; it must remain trackable.") }

if ($errors.Count -gt 0) {
    Write-Host "PHASE 1 CHECK: FAILED" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "PHASE 1 CHECK: PASSED" -ForegroundColor Green
Write-Host "Directories checked: $($requiredDirectories.Count)"
Write-Host "Files checked: $($requiredFiles.Count)"
Write-Host ".env is ignored; .env.example is trackable."
