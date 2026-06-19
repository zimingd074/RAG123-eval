param(
    [string]$OutputDir = $(Join-Path $PSScriptRoot "..\runs\bm25_retrieval_$(Get-Date -Format yyyyMMdd_HHmmss)"),
    [string]$BaseUrl = "http://localhost:9090/api/ragent",
    [string]$Username = "admin",
    [string]$Password = "admin",
    [int]$RequestDelayMs = 250
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
$weak20 = Join-Path $PSScriptRoot "dataset\eval_set_static_weak20_20260613_groundtruth_fixed.jsonl"
$keyword20 = Join-Path $PSScriptRoot "dataset\eval_set_static_keyword20_bm25_20260618.jsonl"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if (-not (Test-Path $python)) { throw "Python venv not found: $python" }
if (-not (Test-Path $weak20)) { throw "Weak-20 dataset not found: $weak20" }
if (-not (Test-Path $keyword20)) { throw "Keyword-heavy dataset not found: $keyword20" }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$env:RAGENT_BASE_URL = $BaseUrl
$env:RAGENT_USERNAME = $Username
$env:RAGENT_PASSWORD = $Password

$jobs = @(
    @{ Label = "bm25_weak20"; Dataset = $weak20; Output = (Join-Path $OutputDir "bm25_weak20.json") },
    @{ Label = "bm25_keyword20"; Dataset = $keyword20; Output = (Join-Path $OutputDir "bm25_keyword20.json") }
)

Push-Location $projectRoot
try {
    foreach ($job in $jobs) {
        Write-Host "=== $($job.Label) ==="
        & $python -m eval.rag.ablation_retrieval `
            --label $job.Label `
            --output $job.Output `
            --dataset $job.Dataset `
            --rrf-k 20 `
            --hnsw-ef-search 200 `
            --keyword-multiplier 2 `
            --request-delay-ms $RequestDelayMs `
            --no-ordinary-fts-conditional
        if ($LASTEXITCODE -ne 0) {
            throw "BM25 retrieval eval failed: $($job.Label)"
        }
    }
} finally {
    Pop-Location
}

Write-Host "output=$OutputDir"