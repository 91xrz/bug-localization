$ErrorActionPreference = 'Stop'
$taskProject = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskDistribution = Get-Content -LiteralPath (Join-Path $taskProject 'configs\elasticsearch\distribution.json') -Raw | ConvertFrom-Json
$taskExpectedJava = Join-Path $taskProject ('.runtime\elasticsearch\elasticsearch-' + $taskDistribution.version + '\jdk\bin\java.exe')
$taskUri = 'http://127.0.0.1:9200'
$taskInfo = Invoke-RestMethod -Uri $taskUri -TimeoutSec 3
if ($taskInfo.cluster_name -ne 'bug-localization-local') { throw 'Unexpected Elasticsearch cluster; refusing to stop it.' }
$taskNodes = Invoke-RestMethod -Uri ($taskUri + '/_nodes/_local/process') -TimeoutSec 3
$taskNode = $taskNodes.nodes.PSObject.Properties.Value | Select-Object -First 1
$taskProcess = Get-Process -Id $taskNode.process.id -ErrorAction Stop
if (-not [string]::Equals($taskProcess.Path, $taskExpectedJava, [StringComparison]::OrdinalIgnoreCase)) { throw 'Node process belongs to another executable; refusing to stop it.' }
Stop-Process -Id $taskNode.process.id
Write-Output 'Stopped the project Elasticsearch node; index data remains in .runtime.'
