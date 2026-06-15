param(
    [Parameter(Mandatory = $true)]
    [string]$RagentJar,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [int]$EmbeddingTimeoutMs = 15000,
    [int]$RequestDelayMs = 250,
    [string]$GroupPattern = ".*"
)

$ErrorActionPreference = "Stop"
$java = "D:\Program Files\Java\jdk-17.0.18\bin\java.exe"
$python = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
$dataset = Join-Path $PSScriptRoot "dataset\eval_set_static_weak20_20260613_groundtruth_fixed.jsonl"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$pathValue = [System.Environment]::GetEnvironmentVariable("Path")
[System.Environment]::SetEnvironmentVariable("Path", $null, [System.EnvironmentVariableTarget]::Process)
[System.Environment]::SetEnvironmentVariable("PATH", $null, [System.EnvironmentVariableTarget]::Process)
[System.Environment]::SetEnvironmentVariable("Path", $pathValue, [System.EnvironmentVariableTarget]::Process)

$env:RAGENT_USERNAME = "admin"
$env:RAGENT_PASSWORD = "admin"
$env:RAGENT_BASE_URL = "http://localhost:9090/api/ragent"

$groups = @(
    @{ label = "k60_m3_always"; k = 60; multiplier = 3; conditional = $false },
    @{ label = "k20_m2_conditional"; k = 20; multiplier = 2; conditional = $true },
    @{ label = "k60_m2_conditional"; k = 60; multiplier = 2; conditional = $true },
    @{ label = "k20_m3_always"; k = 20; multiplier = 3; conditional = $false },
    @{ label = "k60_m3_conditional"; k = 60; multiplier = 3; conditional = $true },
    @{ label = "k20_m2_always"; k = 20; multiplier = 2; conditional = $false },
    @{ label = "k60_m2_always"; k = 60; multiplier = 2; conditional = $false },
    @{ label = "k20_m3_conditional"; k = 20; multiplier = 3; conditional = $true }
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Stop-RagentListener {
    $listener = Get-NetTCPConnection -LocalPort 9090 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($listener) {
        Stop-Process -Id $listener.OwningProcess -Force
        Start-Sleep -Seconds 3
    }
}

function Wait-Ragent {
    param([System.Diagnostics.Process]$Process)
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if ($Process.HasExited) {
            throw "ragent exited with code $($Process.ExitCode)"
        }
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $task = $client.ConnectAsync("127.0.0.1", 9090)
            if ($task.Wait(500) -and $client.Connected) {
                return
            }
        } catch {
            # Service is still starting.
        } finally {
            $client.Dispose()
        }
        Start-Sleep -Seconds 1
    }
    throw "Timed out waiting for ragent on port 9090"
}

foreach ($group in ($groups | Where-Object { $_.label -match $GroupPattern })) {
    Stop-RagentListener
    $label = $group.label
    $stdout = Join-Path $OutputDir "$label.service.stdout.log"
    $stderr = Join-Path $OutputDir "$label.service.stderr.log"
    $arguments = @(
        "-jar", (Resolve-Path $RagentJar).Path,
        "--rag.embedding.timeout-ms=$EmbeddingTimeoutMs",
        "--rag.search.fusion.rrf.k=$($group.k)",
        "--rag.search.channels.keyword-pg.top-k-multiplier=$($group.multiplier)",
        "--rag.search.channels.keyword-pg.ordinary-fts-conditional=$($group.conditional.ToString().ToLowerInvariant())"
    )
    $process = Start-Process `
        -FilePath $java `
        -ArgumentList $arguments `
        -WorkingDirectory (Split-Path (Resolve-Path $RagentJar).Path -Parent) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    Wait-Ragent -Process $process

    Write-Host "=== $label (pid=$($process.Id)) ==="
    $conditionalArg = if ($group.conditional) {
        "--ordinary-fts-conditional"
    } else {
        "--no-ordinary-fts-conditional"
    }
    Push-Location $projectRoot
    try {
        & $python -m eval.rag.ablation_retrieval `
            --label $label `
            --output (Join-Path $OutputDir "$label.json") `
            --dataset $dataset `
            --rrf-k $group.k `
            --keyword-multiplier $group.multiplier `
            --request-delay-ms $RequestDelayMs `
            $conditionalArg
        if ($LASTEXITCODE -ne 0) {
            throw "Ablation failed for $label"
        }
    } finally {
        Pop-Location
    }
}

Write-Host "output=$OutputDir"
