# Start the official Elasticsearch launcher in the current terminal.
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskDistribution = Get-Content -LiteralPath (Join-Path $taskProject 'configs\elasticsearch\distribution.json') -Raw | ConvertFrom-Json
$taskEsHome = Join-Path $taskProject ('.runtime\elasticsearch\elasticsearch-' + $taskDistribution.version)
$taskLauncher = Join-Path $taskEsHome 'bin\elasticsearch.bat'
if (-not (Test-Path -LiteralPath $taskLauncher)) {
    throw 'Run scripts\elasticsearch\setup_elasticsearch.py first.'
}
try { $taskExisting = Invoke-RestMethod -Uri 'http://127.0.0.1:9200' -TimeoutSec 3 } catch { $taskExisting = $null }
if ($null -ne $taskExisting) {
    if ($taskExisting.cluster_name -ne 'bug-localization-local') { throw 'Port 9200 belongs to another cluster.' }
    Write-Output ('Project Elasticsearch is already running: ' + $taskExisting.version.number)
    return
}
$taskSavedConf = $env:ES_PATH_CONF
$taskSavedJava = $env:ES_JAVA_HOME
try {
    $env:ES_PATH_CONF = Join-Path $taskProject '.runtime\elasticsearch\config'
    $env:ES_JAVA_HOME = Join-Path $taskEsHome 'jdk'
    Write-Output 'Starting Elasticsearch at http://127.0.0.1:9200. Keep this terminal open; use Ctrl+C to stop.'
    & $taskLauncher
} finally {
    $env:ES_PATH_CONF = $taskSavedConf
    $env:ES_JAVA_HOME = $taskSavedJava
}
