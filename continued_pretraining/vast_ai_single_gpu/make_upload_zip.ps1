param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$parent = Split-Path -Parent $projectRoot
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $parent "vast_ai_single_gpu_upload.zip"
}
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$allowedPrefix = [System.IO.Path]::GetFullPath($parent).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar

if (-not $resolvedOutput.StartsWith(
        $allowedPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "Output archive must stay inside $parent"
}
if ([System.IO.Path]::GetExtension($resolvedOutput) -ne ".zip") {
    throw "Output archive must have a .zip extension"
}

$relativeItems = @(
    ".gitignore",
    "README.md",
    "WINDOWS_CONNECT_STEP_BY_STEP.md",
    "requirements.txt",
    "benchmark.py",
    "preflight.py",
    "prepare_data.py",
    "train.py",
    "setup.sh",
    "run_experiment.sh",
    "status.sh",
    "pack_results.sh",
    "make_upload_zip.ps1",
    "configs\vast_12gb.yaml",
    "configs\vast_16gb.yaml",
    "configs\vast_24gb.yaml",
    "src\__init__.py",
    "src\config.py",
    "src\data.py",
    "src\runtime.py",
    "tests\test_config.py",
    "tests\test_data.py"
)

$stagingRoot = $projectRoot.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$stagingPath = Join-Path $projectRoot (
    ".qwen-vast-upload-" + [System.Guid]::NewGuid().ToString("N")
)
$stagingPath = [System.IO.Path]::GetFullPath($stagingPath)
$partialOutput = $resolvedOutput + ".partial.zip"

if (-not $stagingPath.StartsWith(
        $stagingRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or -not (Split-Path -Leaf $stagingPath).StartsWith(
        ".qwen-vast-upload-",
        [System.StringComparison]::Ordinal
    )) {
    throw "Unsafe staging path: $stagingPath"
}

New-Item -ItemType Directory -Path $stagingPath | Out-Null
try {
    foreach ($relativeItem in $relativeItems) {
        $source = Join-Path $projectRoot $relativeItem
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Missing upload file: $source"
        }
        $destination = Join-Path $stagingPath $relativeItem
        $destinationParent = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $destinationParent)) {
            New-Item -ItemType Directory -Path $destinationParent -Force |
                Out-Null
        }
        Copy-Item -LiteralPath $source -Destination $destination
    }

    if (Test-Path -LiteralPath $partialOutput) {
        Remove-Item -LiteralPath $partialOutput -Force
    }
    $archiveItems = Get-ChildItem -LiteralPath $stagingPath -Force |
        Select-Object -ExpandProperty FullName
    Compress-Archive -LiteralPath $archiveItems `
        -DestinationPath $partialOutput -CompressionLevel Optimal

    if (Test-Path -LiteralPath $resolvedOutput) {
        Remove-Item -LiteralPath $resolvedOutput -Force
    }
    Move-Item -LiteralPath $partialOutput -Destination $resolvedOutput
}
finally {
    if (Test-Path -LiteralPath $stagingPath) {
        Remove-Item -LiteralPath $stagingPath -Recurse -Force
    }
    if (Test-Path -LiteralPath $partialOutput) {
        Remove-Item -LiteralPath $partialOutput -Force
    }
}

Get-Item -LiteralPath $resolvedOutput | Select-Object FullName, Length
