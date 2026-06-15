param(
    [Parameter(Mandatory = $true)]
    [string]$EmbeddingModel,

    [Parameter(Mandatory = $true)]
    [int]$Dimension,

    [Parameter(Mandatory = $true)]
    [string]$Database,

    [Parameter(Mandatory = $true)]
    [string]$StateDir,

    [string]$PostgresContainer = "postgres",
    [string]$PostgresUser = "postgres",
    [int]$RagentPort = 19090,
    [string]$RagentRepo = "D:\JavaPackage\repository\ragent",
    [string]$Python = "",
    [string]$Dataset = "",
    [int]$EmbeddingTimeoutMs = 5000,
    [int]$RedisDatabase = 11,
    [switch]$ReuseDatabase,
    [switch]$RetryFailedOnly,
    [switch]$RunGoldIntent,
    [switch]$SkipFullChain
)

$ErrorActionPreference = "Stop"

if ($Database -notmatch '^[A-Za-z][A-Za-z0-9_]+$') {
    throw "Database must match ^[A-Za-z][A-Za-z0-9_]+$"
}
if ($Database.ToLowerInvariant() -in @("ragent", "postgres", "template0", "template1")) {
    throw "Refusing to use a shared/current PostgreSQL database: $Database"
}
if ($Dimension -gt 2000) {
    throw "Full-chain ragent validation currently supports vector HNSW up to 2000 dimensions. Use embedding-benchmark pgvector validation for halfvec/4096 experiments."
}
if ($RedisDatabase -lt 1 -or $RedisDatabase -gt 15) {
    throw "RedisDatabase must be an isolated database number from 1 to 15."
}
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
}
if (-not $Dataset) {
    $Dataset = Join-Path $PSScriptRoot "dataset\eval_set_v1_all.jsonl"
}

$resolvedStateDir = [System.IO.Path]::GetFullPath($StateDir)
$resolvedRagentRepo = [System.IO.Path]::GetFullPath($RagentRepo)
$schemaPath = Join-Path $resolvedRagentRepo "resources\database\schema_pg.sql"
$dataPath = Join-Path $resolvedRagentRepo "resources\database\init_data_pg.sql"
$logOut = Join-Path $resolvedStateDir "ragent.stdout.log"
$logErr = Join-Path $resolvedStateDir "ragent.stderr.log"
New-Item -ItemType Directory -Force -Path $resolvedStateDir | Out-Null
$collectionPrefix = ($Database.ToLowerInvariant() -replace '_', '-')
if ($collectionPrefix.Length -gt 36) {
    $collectionPrefix = $collectionPrefix.Substring(0, 36).TrimEnd('-')
}

$existsOutput = & docker exec $PostgresContainer psql -U $PostgresUser -d postgres -Atc "SELECT 1 FROM pg_database WHERE datname='$Database';"
$exists = if ($null -eq $existsOutput) { "" } else { "$existsOutput".Trim() }
if ($exists -eq "1" -and -not $ReuseDatabase) {
    throw "Database '$Database' already exists. Use a fresh name or pass -ReuseDatabase explicitly."
}
if ($exists -ne "1") {
    & docker exec $PostgresContainer createdb -U $PostgresUser $Database
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create PostgreSQL database '$Database'"
    }
}

if (-not $ReuseDatabase) {
    $schemaSql = (Get-Content -Raw -Encoding UTF8 $schemaPath) -replace 'vector\(1536\)', "vector($Dimension)"
    $schemaSql | & docker exec -i $PostgresContainer psql -v ON_ERROR_STOP=1 -U $PostgresUser -d $Database
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize schema in '$Database'"
    }
    Get-Content -Raw -Encoding UTF8 $dataPath | & docker exec -i $PostgresContainer psql -v ON_ERROR_STOP=1 -U $PostgresUser -d $Database
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize seed data in '$Database'"
    }
}

$env:SPRING_DATASOURCE_URL = "jdbc:postgresql://127.0.0.1:5432/$Database`?client_encoding=UTF8"
$env:RAG_DEFAULT_DIMENSION = "$Dimension"
$env:RAG_EMBEDDING_MODEL_ID = $EmbeddingModel
$env:RAG_EMBEDDING_TIMEOUT_MS = "$EmbeddingTimeoutMs"
$env:SPRING_DATA_REDIS_DATABASE = "$RedisDatabase"
$env:SERVER_PORT = "$RagentPort"
$env:PYTHONUTF8 = "1"
$env:UNIQUE_NAME = "-$collectionPrefix"
$env:QWEN_EMBEDDING_4B_ENABLED = if ($EmbeddingModel -eq "qwen-emb-4b") { "true" } else { "false" }
$env:QWEN_EMBEDDING_06B_ENABLED = if ($EmbeddingModel -eq "qwen-emb-06b") { "true" } else { "false" }
$env:QWEN_EMBEDDING_06B_DIMENSION = "$Dimension"
$env:BGE_M3_ENABLED = if ($EmbeddingModel -eq "bge-m3") { "true" } else { "false" }
$env:BGE_LARGE_ZH_ENABLED = if ($EmbeddingModel -eq "bge-large-zh") { "true" } else { "false" }

& docker exec redis redis-cli -a 123456 -n $RedisDatabase FLUSHDB | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to clear isolated Redis database $RedisDatabase"
}

$process = Start-Process `
    -FilePath (Join-Path $resolvedRagentRepo "mvnw.cmd") `
    -ArgumentList "-f", "bootstrap\pom.xml", "-Dspotless.apply.skip=true", "spring-boot:run" `
    -WorkingDirectory $resolvedRagentRepo `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -PassThru

try {
    $env:RAGENT_BASE_URL = "http://localhost:$RagentPort/api/ragent"
    $deadline = (Get-Date).AddMinutes(3)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($process.HasExited) {
            throw "ragent exited during startup. See $logErr"
        }
        try {
            $body = @{
                username = $env:RAGENT_USERNAME
                password = $env:RAGENT_PASSWORD
            } | ConvertTo-Json
            $response = Invoke-RestMethod `
                -Uri "$env:RAGENT_BASE_URL/auth/login" `
                -Method Post `
                -ContentType "application/json" `
                -Body $body `
                -TimeoutSec 5
            if ($response.data.token) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ready) {
        throw "ragent did not become ready within 3 minutes. See $logErr"
    }

    if (-not (Test-Path (Join-Path $resolvedStateDir "kb_ids.json"))) {
        & $Python (Join-Path $PSScriptRoot "init\create_kbs.py") `
            --embedding-model $EmbeddingModel `
            --dimension $Dimension `
            --collection-prefix $collectionPrefix `
            --state-dir $resolvedStateDir
        if ($LASTEXITCODE -ne 0) { throw "create_kbs.py failed" }
    } else {
        Write-Host "Reuse existing kb_ids.json"
    }

    if ($RetryFailedOnly) {
        $failedIds = @(
            & docker exec $PostgresContainer psql -U $PostgresUser -d $Database -Atc `
                "SELECT id FROM t_knowledge_document WHERE deleted=0 AND status='failed' ORDER BY id;"
        )
        Write-Host "Retrying $($failedIds.Count) failed documents sequentially"
        $headers = @{ Authorization = $response.data.token }
        foreach ($docId in $failedIds) {
            $completed = $false
            for ($attempt = 1; $attempt -le 3 -and -not $completed; $attempt++) {
                Invoke-RestMethod `
                    -Uri "$env:RAGENT_BASE_URL/knowledge-base/docs/$docId/chunk" `
                    -Method Post `
                    -Headers $headers `
                    -TimeoutSec 15 | Out-Null
                $docDeadline = (Get-Date).AddMinutes(2)
                while ((Get-Date) -lt $docDeadline) {
                    $status = & docker exec $PostgresContainer psql -U $PostgresUser -d $Database -Atc `
                        "SELECT status FROM t_knowledge_document WHERE id='$docId';"
                    $status = "$status".Trim()
                    if ($status -eq "success") {
                        $completed = $true
                        Write-Host "Document $docId succeeded on attempt $attempt"
                        break
                    }
                    if ($status -eq "failed") {
                        Write-Host "Document $docId failed on attempt $attempt"
                        break
                    }
                    Start-Sleep -Seconds 2
                }
            }
            if (-not $completed) {
                throw "Document $docId still failed after 3 sequential retries"
            }
        }
    } else {
        $uploadArgs = @(
            (Join-Path $PSScriptRoot "init\upload_docs.py"),
            "--state-dir", $resolvedStateDir
        )
        if ($ReuseDatabase -and (Test-Path (Join-Path $resolvedStateDir "doc_id_map.json"))) {
            $resetSql = "UPDATE t_knowledge_document SET status='failed' WHERE status='running';"
            & docker exec $PostgresContainer psql -v ON_ERROR_STOP=1 -U $PostgresUser -d $Database -c $resetSql
            if ($LASTEXITCODE -ne 0) { throw "Failed to reset interrupted documents in '$Database'" }
            $uploadArgs += "--rechunk-existing"
        }
        & $Python @uploadArgs
        if ($LASTEXITCODE -ne 0) { throw "upload_docs.py failed" }
    }

    $ingestionDeadline = (Get-Date).AddMinutes(20)
    $ingestionReady = $false
    while ((Get-Date) -lt $ingestionDeadline) {
        $statsSql = @"
SELECT
    count(*) FILTER (WHERE status='success'),
    count(*),
    coalesce(sum(chunk_count), 0),
    (SELECT count(*) FROM t_knowledge_vector)
FROM t_knowledge_document
WHERE deleted=0;
"@
        $statsOutput = & docker exec $PostgresContainer psql -U $PostgresUser -d $Database -At -F "|" -c $statsSql
        if ($LASTEXITCODE -ne 0) { throw "Failed to inspect ingestion state in '$Database'" }
        $stats = "$statsOutput".Trim().Split("|")
        if ($stats.Count -eq 4) {
            $successDocs = [int]$stats[0]
            $totalDocs = [int]$stats[1]
            $chunkCount = [int]$stats[2]
            $vectorCount = [int]$stats[3]
            Write-Host "Ingestion: success=$successDocs/$totalDocs chunks=$chunkCount vectors=$vectorCount"
            if ($totalDocs -eq 115 -and $successDocs -eq $totalDocs -and
                $chunkCount -gt 0 -and $chunkCount -eq $vectorCount) {
                $ingestionReady = $true
                break
            }
        }
        if ($process.HasExited) {
            throw "ragent exited during ingestion. See $logErr"
        }
        Start-Sleep -Seconds 5
    }
    if (-not $ingestionReady) {
        throw "Document ingestion did not complete within 20 minutes in '$Database'"
    }

    & $Python (Join-Path $PSScriptRoot "init\build_intent_tree.py") `
        --state-dir $resolvedStateDir `
        --dataset $Dataset
    if ($LASTEXITCODE -ne 0) { throw "build_intent_tree.py failed" }

    if ($RunGoldIntent) {
        & $Python -m eval.rag.gold_intent_retrieval `
            --state-dir $resolvedStateDir `
            --dataset $Dataset
        if ($LASTEXITCODE -ne 0) { throw "Gold-intent retrieval evaluation failed" }
    }

    if (-not $SkipFullChain) {
        & $Python -m eval rag run `
            --state-dir $resolvedStateDir `
            --embedding-model $EmbeddingModel `
            --dimension $Dimension `
            --dataset $Dataset `
            --profile static-v1
        if ($LASTEXITCODE -ne 0) { throw "Full-chain evaluation failed" }
    }
} finally {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id
        $process.WaitForExit(10000)
    }
    $listener = Get-NetTCPConnection -LocalPort $RagentPort -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        $listener |
            Select-Object -ExpandProperty OwningProcess -Unique |
            ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    }
}
