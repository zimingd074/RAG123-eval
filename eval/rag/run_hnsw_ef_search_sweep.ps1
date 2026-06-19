param(
    [Parameter(Mandatory = $true)]
    [string]$RagentJar,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [int[]]$EfSearchValues = @(40, 120, 200, 400),
    [int]$Rounds = 1,
    [int]$RequestDelayMs = 250,
    [int]$Port = 9090,
    [string]$Java = "java",
    [string]$Python = "",
    [string]$Dataset = "",
    [AllowEmptyString()]
    [string]$RedisPassword = "123456",
    [int]$RrfK = 20,
    [int]$KeywordMultiplier = 2,
    [switch]$NoOrdinaryFtsConditional
)

$ErrorActionPreference = "Stop"

if ($Rounds -lt 1) {
    throw "Rounds must be at least 1."
}
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
}
if (-not $Dataset) {
    $Dataset = Join-Path $PSScriptRoot "dataset\eval_set_static_weak20_20260613_groundtruth_fixed.jsonl"
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$resolvedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null

$env:RAGENT_USERNAME = "admin"
$env:RAGENT_PASSWORD = "admin"
$env:RAGENT_BASE_URL = "http://localhost:$Port/api/ragent"

function Stop-RagentListener {
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($listener) {
        Stop-Process -Id $listener.OwningProcess -Force
        Start-Sleep -Seconds 3
    }
}

function Wait-Ragent {
    param([System.Diagnostics.Process]$Process)
    $deadline = (Get-Date).AddMinutes(3)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            throw "ragent exited with code $($Process.ExitCode)"
        }
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $task = $client.ConnectAsync("127.0.0.1", $Port)
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
    throw "Timed out waiting for ragent on port $Port"
}

function Get-MissedCases {
    param($Result)
    $misses = @()
    foreach ($sample in $Result.samples) {
        if ([double]$sample.recall5 -lt 1.0) {
            $misses += $sample.query_id
        }
    }
    return ($misses -join ";")
}

function Get-NumberOrZero {
    param($Value)
    if ($null -eq $Value) {
        return 0
    }
    return [double]$Value
}

function Write-SummaryArtifacts {
    param([array]$Rows)

    $csv = Join-Path $resolvedOutputDir "hnsw_ef_search_summary.csv"
    $md = Join-Path $resolvedOutputDir "hnsw_ef_search_summary.md"
    $Rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv

    $lines = @(
        "# HNSW ef_search Weak-20 Sweep",
        "",
        "Dataset: ``$Dataset``",
        "",
        "| ef_search | round | hit@5 | recall@5 | mrr@10 | vector_search_p95_ms | retrieval_p95_ms | wall_p95_ms | missed_cases |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    foreach ($row in $Rows) {
        $lines += "| $($row.ef_search) | $($row.round) | $($row.hit5) | $($row.recall5) | $($row.mrr10) | $($row.vector_search_p95_ms) | $($row.retrieval_p95_ms) | $($row.wall_p95_ms) | $($row.missed_cases) |"
    }

    $byEf = $Rows | Group-Object ef_search | ForEach-Object {
        $items = $_.Group
        [PSCustomObject]@{
            ef_search = [int]$_.Name
            hit5 = [Math]::Round(($items | Measure-Object hit5 -Average).Average, 6)
            recall5 = [Math]::Round(($items | Measure-Object recall5 -Average).Average, 6)
            mrr10 = [Math]::Round(($items | Measure-Object mrr10 -Average).Average, 6)
            vector_search_p95_ms = [Math]::Round(($items | Measure-Object vector_search_p95_ms -Average).Average, 2)
        }
    } | Sort-Object ef_search

    $ef120 = $byEf | Where-Object { $_.ef_search -eq 120 } | Select-Object -First 1
    $ef200 = $byEf | Where-Object { $_.ef_search -eq 200 } | Select-Object -First 1
    $ef400 = $byEf | Where-Object { $_.ef_search -eq 400 } | Select-Object -First 1
    $decision = "Manual review required: expected ef_search values 120, 200, and 400 were not all present."
    $hasErrors = ($Rows | Where-Object { [int]$_.error_count -gt 0 } | Select-Object -First 1) -ne $null
    if ($hasErrors) {
        $decision = "Invalid run: at least one ef_search group returned errors. Fix service/runtime errors and rerun."
    } elseif ($ef120 -and $ef200 -and $ef400) {
        $recallGap400 = [double]$ef400.recall5 - [double]$ef200.recall5
        $recallGap120 = [double]$ef200.recall5 - [double]$ef120.recall5
        $latencyDrop120 = 0.0
        if ([double]$ef200.vector_search_p95_ms -gt 0) {
            $latencyDrop120 = ([double]$ef200.vector_search_p95_ms - [double]$ef120.vector_search_p95_ms) / [double]$ef200.vector_search_p95_ms
        }

        if ($recallGap400 -gt 0.01) {
            $decision = "Consider raising to 400: ef_search=400 improved recall@5 by more than 1 percentage point over 200."
        } elseif ([Math]::Abs($recallGap120) -lt 0.000001 -and $latencyDrop120 -ge 0.10) {
            $decision = "Consider lowering to 120: recall matched 200 and vector p95 latency improved by at least 10%."
        } else {
            $decision = "Keep 200: it is close to the 400 recall ceiling without the high-ef latency tradeoff."
        }
    }
    $lines += ""
    $lines += "## Decision"
    $lines += ""
    $lines += $decision
    $lines += ""
    $lines += "CSV: ``$csv``"
    $lines | Set-Content -Encoding UTF8 -Path $md
    Write-Host "summary_csv=$csv"
    Write-Host "summary_md=$md"
}

$rows = @()
try {
    foreach ($efSearch in $EfSearchValues) {
        foreach ($round in 1..$Rounds) {
            Stop-RagentListener
            $label = "ef$($efSearch)_r$($round)"
            $stdout = Join-Path $resolvedOutputDir "$label.service.stdout.log"
            $stderr = Join-Path $resolvedOutputDir "$label.service.stderr.log"
            $arguments = @(
                "-jar", (Resolve-Path $RagentJar).Path,
                "--server.port=$Port",
                "--app.eval.enabled=true",
                "--spring.data.redis.password=$RedisPassword",
                "--rag.vector.pg.hnsw-ef-search=$efSearch"
            )
            $process = Start-Process `
                -FilePath $Java `
                -ArgumentList $arguments `
                -WorkingDirectory (Split-Path (Resolve-Path $RagentJar).Path -Parent) `
                -WindowStyle Hidden `
                -RedirectStandardOutput $stdout `
                -RedirectStandardError $stderr `
                -PassThru
            Wait-Ragent -Process $process

            Write-Host "=== $label (pid=$($process.Id)) ==="
            $conditionalArg = if ($NoOrdinaryFtsConditional) {
                "--no-ordinary-fts-conditional"
            } else {
                "--ordinary-fts-conditional"
            }
            $jsonOutput = Join-Path $resolvedOutputDir "$label.json"
            Push-Location $projectRoot
            try {
                & $Python -m eval.rag.ablation_retrieval `
                    --label $label `
                    --output $jsonOutput `
                    --dataset $Dataset `
                    --rrf-k $RrfK `
                    --keyword-multiplier $KeywordMultiplier `
                    --hnsw-ef-search $efSearch `
                    --request-delay-ms $RequestDelayMs `
                    $conditionalArg
                if ($LASTEXITCODE -ne 0) {
                    throw "Ablation failed for $label"
                }
            } finally {
                Pop-Location
            }

            $result = Get-Content -Raw -Encoding UTF8 -Path $jsonOutput | ConvertFrom-Json
            $summary = $result.summary
            $rows += [PSCustomObject]@{
                ef_search = $efSearch
                round = $round
                sample_count = $summary.sample_count
                error_count = $summary.error_count
                hit5 = [Math]::Round([double]$summary.hit5, 6)
                recall5 = [Math]::Round([double]$summary.recall5, 6)
                mrr10 = [Math]::Round([double]$summary.mrr10, 6)
                vector_search_p95_ms = [Math]::Round((Get-NumberOrZero $summary.vector_search_p95_ms), 2)
                retrieval_p95_ms = [Math]::Round((Get-NumberOrZero $summary.retrieval_p95_ms), 2)
                wall_p95_ms = [Math]::Round((Get-NumberOrZero $summary.wall_p95_ms), 2)
                missed_cases = Get-MissedCases -Result $result
                result_json = $jsonOutput
            }
        }
    }
} finally {
    Stop-RagentListener
}

Write-SummaryArtifacts -Rows $rows
Write-Host "output=$resolvedOutputDir"
