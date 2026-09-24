param([string]$PythonPath = '', [switch]$Detached)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $PythonPath) { $PythonPath = Join-Path $root 'workspace\sd-scripts\venv_5050\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'Python environment not found; specify -PythonPath.' }
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
if ($Detached) {
    $windowlessPython = Join-Path (Split-Path $PythonPath -Parent) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $windowlessPython -PathType Leaf)) { throw 'pythonw.exe not found beside the selected Python environment.' }
    $logDirectory = Join-Path $PSScriptRoot 'data\launcher'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $logName = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N')
    Start-Process -FilePath $windowlessPython -ArgumentList @('-m', 'desktop_agent.app') -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $logDirectory ($logName + '.stdout.log')) `
        -RedirectStandardError (Join-Path $logDirectory ($logName + '.stderr.log')) | Out-Null
    return
}
Push-Location $root
try {
    & $PythonPath -m desktop_agent.app
    if ($LASTEXITCODE -ne 0) { throw "Local Desk exited with code $LASTEXITCODE" }
} finally { Pop-Location }