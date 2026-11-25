param(
    [Parameter(Mandatory = $true)][string]$GameExe,
    [string]$FrametimeDump,
    [ValidateRange(1, 32)][int]$CopyWorkers = [Math]::Max(2, [Math]::Min(8, [Environment]::ProcessorCount / 2)),
    [ValidateRange(1, 64)][int]$PipelineWorkers = [Math]::Max(2, [Math]::Min(8, [Environment]::ProcessorCount / 2)),
    [ValidateSet(64,128,256,512,1024,2048,4096)][int]$WindowKiB = 4096,
    [switch]$NoMeshCompaction
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$binary = Join-Path $repo 'target\release\victory.exe'
$game = (Resolve-Path -LiteralPath $GameExe).Path
if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
    throw 'Build the optimized executable first: cargo build --release -p victory'
}
if (-not $FrametimeDump) {
    $folder = Join-Path (Split-Path -Parent $game) 'victory-performance'
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    $FrametimeDump = Join-Path $folder 'frametime.csv'
}
# Only this invocation uses the performance settings; restore the shell on exit.
$settings = @{
    VICTORY_FRAMETIME_DUMP = $FrametimeDump
    VICTORY_COPY_WORKERS = "$CopyWorkers"
    VICTORY_PIPELINE_WORKERS = "$PipelineWorkers"
    VICTORY_PIPELINE_PRIORITY = 'normal'
    VICTORY_WINDOW_KB = "$WindowKiB"
    VICTORY_WINDOW_TAILS = $(if ($WindowKiB -lt 4096) { '1' } else { '0' })
    VICTORY_CARD_FETCH_INFLIGHT = '1'
    VICTORY_WIDE_MERGE = '1'
    VICTORY_CARD_REHASH_RUNS = '1'
    VICTORY_JOIN_COPIES = '1'
    VICTORY_MESH_COMPACT = $(if ($NoMeshCompaction) { '0' } else { '1' })
    VICTORY_D3D12_DEBUG = $null
    VICTORY_CAPTURE = $null
    VICTORY_CAPTURE_WORLD = $null
    VICTORY_CAPTURE_FRAMES = $null
    VICTORY_CAPTURE_MENUS = $null
    VICTORY_CAPTURE_WORLD_ALSO = $null
    VICTORY_KEYS = $null
    VICTORY_GPU_TIMING = $null
}
$previous = @{}
try {
    foreach ($key in $settings.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    Push-Location -LiteralPath $repo
    try { & $binary run $game --execute --console series-s }
    finally { Pop-Location }
}
finally {
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}
